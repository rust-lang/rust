;;; ra.el --- Rust analyzer emacs bindings -*- lexical-binding: t; -*-
;;; Commentary:
;;; Small utilities for interacting with Rust analyzer.
;;; Run
;;;   cargo install --git https://github.com/matklad/rust-analyzer/ --bin ra_cli
;;; to install the binary, copy-paste the bellow code to your `.init.el` and use
;;; `ra-extend-selection` and `ra-shrink-selection` functions
;;; Code:


(defvar ra--selections-cache '(0 0 ()))
(defun ra--cache-tick ()
  "Get buffer modification count for cache."
  (nth 0 ra--selections-cache))
(defun ra--cache-sel ()
  "Get current selection for cache."
  (nth 1 ra--selections-cache))
(defun ra--cache-nth-sel (n)
  "Get Nth selection."
  (nth n (nth 2 ra--selections-cache)))
(defun ra--cache-set-nth-sel (n)
  "Get Nth selection."
  (setf (nth 1 ra--selections-cache) n)
  (nth n (nth 2 ra--selections-cache)))


(defun ra-extend-selection ()
  "Extend START END region to contain the encompassing syntactic construct."
  (interactive)
  (let* ((p (point))
         (m (or (and mark-active (mark)) p))
         (start (min p m))
         (end (max p m)))
    (ra--extend-selection start end)))


(defun ra-shrink-selection (start end)
  "Shrink START END region to contain previous selection."
  (interactive "r")
  (ra--freshen-cache start end)
  (let ((sel-id (ra--cache-sel)))
    (if (not (= 0 sel-id))
        (let* ((r (ra--cache-set-nth-sel (- sel-id 1))))
          (push-mark (nth 0 r) t t)
          (goto-char (nth 1 r))
          (setq deactivate-mark nil)))))

; Add this to setup keybinding
; (require 'rust-mode)
; (define-key rust-mode-map (kbd "C-w") 'ra-extend-selection)
; (define-key rust-mode-map (kbd "C-S-w") 'ra-shrink-selection)



(defun ra--extend-selection (start end)
  "Extend START END region to contain the encompassing syntactic construct."
  (ra--freshen-cache start end)
  (let* ((next-sel-idx (+ 1 (ra--cache-sel)))
         (r (ra--cache-set-nth-sel next-sel-idx)))
    (push-mark (nth 0 r) t t)
    (goto-char (nth 1 r))
    (setq deactivate-mark nil)))

(defun ra--selections (start end)
  "Get list of selections for START END from Rust analyzer."
  (read (with-output-to-string
          (call-process-region
           (point-min) (point-max)
           "ra_cli" nil standard-output nil
           "extend-selection"
           (number-to-string start)
           (number-to-string end)))))

(defun ra--freshen-cache (start end)
  "Make selection cache up-to-date for current buffer state and START END."
  (if (not (and
            (= (buffer-modified-tick) (ra--cache-tick))
            (equal `(,start ,end) (ra--cache-nth-sel (ra--cache-sel)))))
      (ra--set-cache start end)))

(defun ra--set-cache (start end)
  "Set selections cache for current buffer state and START END."
  (setq ra--selections-cache `(,(buffer-modified-tick) 0 ,(ra--selections start end))))

(provide 'ra)
;;; ra.el ends here
