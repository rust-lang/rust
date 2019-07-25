;;; ra-emacs-lsp.el --- Rust analyzer emacs bindings for emacs-lsp -*- lexical-binding: t; -*-
;;; Code:

(require 'lsp)
(require 'dash)
(require 'ht)

;; This currently
;;  - sets up rust-analyzer with emacs-lsp, giving
;;    - code actions
;;    - completion (use company-lsp for proper snippet support)
;;    - imenu support
;;    - on-type formatting
;;    - 'hover' type information & documentation (with lsp-ui)
;;  - implements source changes (for code actions etc.), except for file system changes
;;  - implements joinLines (you need to bind rust-analyzer-join-lines to a key)
;;  - implements extendSelection (either bind rust-analyzer-extend-selection to a key, or use expand-region)
;;  - provides rust-analyzer-inlay-hints-mode for inline type hints

;; What's missing:
;;  - file system changes in apply-source-change
;;  - semantic highlighting
;;  - onEnter, parentModule, findMatchingBrace
;;  - runnables
;;  - the debugging commands (syntaxTree and analyzerStatus)
;;  - more

;; Also, there's a problem with company-lsp's caching being too eager, sometimes
;; resulting in outdated completions.

(defcustom rust-analyzer-command '("ra_lsp_server")
  ""
  :type '(repeat (string)))

(defconst rust-analyzer--notification-handlers
  '(("rust-analyzer/publishDecorations" . (lambda (_w _p)))))

(defconst rust-analyzer--action-handlers
  '(("rust-analyzer.applySourceChange" .
     (lambda (p) (rust-analyzer--apply-source-change-command p)))))

(defun rust-analyzer--uri-filename (text-document)
  (lsp--uri-to-path (gethash "uri" text-document)))

(defun rust-analyzer--goto-lsp-loc (loc)
  (-let (((&hash "line" "character") loc))
    (goto-line (1+ line))
    (move-to-column character)))

(defun rust-analyzer--apply-text-document-edit (edit)
  "Like lsp--apply-text-document-edit, but it allows nil version."
  (let* ((ident (gethash "textDocument" edit))
         (filename (rust-analyzer--uri-filename ident))
         (version (gethash "version" ident)))
    (with-current-buffer (find-file-noselect filename)
      (when (or (not version) (= version (lsp--cur-file-version)))
        (lsp--apply-text-edits (gethash "edits" edit))))))

(defun rust-analyzer--apply-source-change (data)
  ;; TODO fileSystemEdits
  (seq-doseq (it (-> data (ht-get "workspaceEdit") (ht-get "documentChanges")))
    (rust-analyzer--apply-text-document-edit it))
  (-when-let (cursor-position (ht-get data "cursorPosition"))
    (let ((filename (rust-analyzer--uri-filename (ht-get cursor-position "textDocument")))
          (position (ht-get cursor-position "position")))
      (find-file filename)
      (rust-analyzer--goto-lsp-loc position))))

(defun rust-analyzer--apply-source-change-command (p)
  (let ((data (-> p (ht-get "arguments") (seq-first))))
    (rust-analyzer--apply-source-change data)))

(lsp-register-client
 (make-lsp-client
  :new-connection (lsp-stdio-connection (lambda () rust-analyzer-command))
  :notification-handlers (ht<-alist rust-analyzer--notification-handlers)
  :action-handlers (ht<-alist rust-analyzer--action-handlers)
  :major-modes '(rust-mode)
  :ignore-messages nil
  :server-id 'rust-analyzer))

(with-eval-after-load 'company-lsp
  ;; company-lsp provides a snippet handler for rust by default that adds () after function calls, which RA does better
  (setq company-lsp--snippet-functions (cl-delete "rust" company-lsp--snippet-functions :key #'car :test #'equal)))

;; join lines

(defun rust-analyzer--join-lines-params ()
  "Join lines params."
  (list :textDocument (lsp--text-document-identifier)
        :range (if (use-region-p)
                   (lsp--region-to-range (region-beginning) (region-end))
                 (lsp--region-to-range (point) (point)))))

(defun rust-analyzer-join-lines ()
  (interactive)
  (->
   (lsp-send-request (lsp-make-request "rust-analyzer/joinLines"
                                       (rust-analyzer--join-lines-params)))
   (rust-analyzer--apply-source-change)))

;; extend selection

(defun rust-analyzer-extend-selection ()
  (interactive)
  (-let (((&hash "start" "end") (rust-analyzer--extend-selection)))
    (rust-analyzer--goto-lsp-loc start)
    (set-mark (point))
    (rust-analyzer--goto-lsp-loc end)
    (exchange-point-and-mark)))

(defun rust-analyzer--extend-selection-params ()
  "Extend selection params."
  (list :textDocument (lsp--text-document-identifier)
        :selections
        (vector
         (if (use-region-p)
             (lsp--region-to-range (region-beginning) (region-end))
           (lsp--region-to-range (point) (point))))))

(defun rust-analyzer--extend-selection ()
  (->
   (lsp-send-request
    (lsp-make-request
     "rust-analyzer/extendSelection"
     (rust-analyzer--extend-selection-params)))
   (ht-get "selections")
   (seq-first)))

(defun rust-analyzer--add-er-expansion ()
  (make-variable-buffer-local 'er/try-expand-list)
  (setq er/try-expand-list (append
                            er/try-expand-list
                            '(rust-analyzer-extend-selection))))

(with-eval-after-load 'expand-region
  ;; add the expansion for all existing rust-mode buffers. If expand-region is
  ;; loaded lazily, it might be loaded when the first rust buffer is opened, and
  ;; then it's too late for the hook for that buffer
  (dolist (buf (buffer-list))
    (with-current-buffer buf
      (when (eq 'rust-mode major-mode)
        (rust-analyzer--add-er-expansion))))
  (add-hook 'rust-mode-hook 'rust-analyzer--add-er-expansion))

;; runnables
(defvar rust-analyzer--last-runnable nil)

(defun rust-analyzer--runnables-params ()
  (list :textDocument (lsp--text-document-identifier)
        :position (lsp--cur-position)))

(defun rust-analyzer--runnables ()
  (lsp-send-request (lsp-make-request "rust-analyzer/runnables"
                                      (rust-analyzer--runnables-params))))

(defun rust-analyzer--select-runnable ()
  (lsp--completing-read
   "Select runnable:"
   (if rust-analyzer--last-runnable
       (cons rust-analyzer--last-runnable (rust-analyzer--runnables))
       (rust-analyzer--runnables))
   (-lambda ((&hash "label")) label)))

(defun rust-analyzer-run (runnable)
  (interactive (list (rust-analyzer--select-runnable)))
  (-let (((&hash "env" "bin" "args" "label") runnable))
    (compilation-start
     (string-join (append (list bin) args '()) " ")
     ;; cargo-process-mode is nice, but try to work without it...
     (if (functionp 'cargo-process-mode) 'cargo-process-mode nil)
     (lambda (_) (concat "*" label "*")))
    (setq rust-analyzer--last-runnable runnable)))

(defun rust-analyzer-rerun (&optional runnable)
  (interactive (list (or rust-analyzer--last-runnable
                         (rust-analyzer--select-runnable))))
  (rust-analyzer-run (or runnable rust-analyzer--last-runnable)))

;; analyzer status buffer
(define-derived-mode rust-analyzer-status-mode special-mode "Rust-Analyzer-Status"
  "Mode for the rust-analyzer status buffer.")

(defvar-local rust-analyzer--status-buffer-workspace nil)

(defun rust-analyzer-status ()
  "Displays status information for rust-analyzer."
  (interactive)
  (let* ((workspace (lsp-find-workspace 'rust-analyzer (buffer-file-name)))
         (buf (get-buffer-create (concat "*rust-analyzer status " (with-lsp-workspace workspace (lsp-workspace-root)) "*"))))
    (with-current-buffer buf
      (rust-analyzer-status-mode)
      (setq rust-analyzer--status-buffer-workspace workspace)
      (rust-analyzer-status-buffer-refresh))
    (pop-to-buffer buf)))

(defun rust-analyzer-status-buffer-refresh ()
  (interactive)
  (when rust-analyzer--status-buffer-workspace
    (let ((inhibit-read-only t))
      (erase-buffer)
      (insert (with-lsp-workspace rust-analyzer--status-buffer-workspace
                (lsp-send-request (lsp-make-request
                                   "rust-analyzer/analyzerStatus")))))))


(defun rust-analyzer--syntax-tree-params ()
  "Syntax tree params."
  (list :textDocument (lsp--text-document-identifier)
        :range (if (use-region-p)
                   (lsp--region-to-range (region-beginning) (region-end))
                 (lsp--region-to-range (point-min) (point-max)))))

(defun rust-analyzer-syntax-tree ()
  "Displays syntax tree for current buffer."
  (interactive)
  (when (eq 'rust-mode major-mode)
    (let* ((workspace (lsp-find-workspace 'rust-analyzer (buffer-file-name)))
           (buf (get-buffer-create (concat "*rust-analyzer syntax tree " (with-lsp-workspace workspace (lsp-workspace-root)) "*"))))
      (when workspace
        (let ((parse-result (with-lsp-workspace workspace
                              (lsp-send-request (lsp-make-request
                                                 "rust-analyzer/syntaxTree"
                                                 (rust-analyzer--syntax-tree-params))))))
          (with-current-buffer buf
            (let ((inhibit-read-only t))
              (erase-buffer)
              (insert parse-result)))
          (pop-to-buffer buf))))))

;; inlay hints
(defun rust-analyzer--update-inlay-hints ()
  (lsp-send-request-async
   (lsp-make-request "rust-analyzer/inlayHints"
                     (list :textDocument (lsp--text-document-identifier)))
   (lambda (res)
     (remove-overlays (point-min) (point-max) 'rust-analyzer--inlay-hint t)
     (dolist (hint res)
       (-let* (((&hash "range" "label" "kind") hint)
               ((beg . end) (lsp--range-to-region range))
               (overlay (make-overlay beg end)))
         (overlay-put overlay 'rust-analyzer--inlay-hint t)
         (overlay-put overlay 'evaporate t)
         (overlay-put overlay 'after-string (propertize (concat ": " label)
                                                        'font-lock-face 'font-lock-comment-face)))))
   'tick)
  nil)

(defvar-local rust-analyzer--inlay-hints-timer nil)

(defun rust-analyzer--inlay-hints-change-handler (&rest rest)
  (when rust-analyzer--inlay-hints-timer
    (cancel-timer rust-analyzer--inlay-hints-timer))
  (setq rust-analyzer--inlay-hints-timer
        (run-with-idle-timer 0.1 nil #'rust-analyzer--update-inlay-hints)))

(define-minor-mode rust-analyzer-inlay-hints-mode
  "Mode for showing inlay hints."
  nil nil nil
  (cond
   (rust-analyzer-inlay-hints-mode
    (rust-analyzer--update-inlay-hints)
    (add-hook 'after-change-functions #'rust-analyzer--inlay-hints-change-handler nil t))
   (t
    (remove-overlays (point-min) (point-max) 'rust-analyzer--inlay-hint t)
    (remove-hook 'after-change-functions #'rust-analyzer--inlay-hints-change-handler t))))


(provide 'ra-emacs-lsp)
;;; ra-emacs-lsp.el ends here
