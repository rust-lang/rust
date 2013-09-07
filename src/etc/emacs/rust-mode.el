;;; rust-mode.el --- A major emacs mode for editing Rust source code

;; Version: 0.2.0
;; Author: Mozilla
;; Url: https://github.com/mozilla/rust

(eval-when-compile (require 'cl))
(eval-when-compile (require 'misc))

;; Syntax definitions and helpers
(defvar rust-mode-syntax-table
  (let ((table (make-syntax-table)))

    ;; Operators
    (loop for i in '(?+ ?- ?* ?/ ?& ?| ?^ ?! ?< ?> ?~ ?@)
          do (modify-syntax-entry i "." table))

    ;; Strings
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?\\ "\\" table)

    ;; _ is a word-char
    (modify-syntax-entry ?_ "w" table)

    ;; Comments
    (modify-syntax-entry ?/  ". 124b" table)
    (modify-syntax-entry ?*  ". 23"   table)
    (modify-syntax-entry ?\n "> b"    table)
    (modify-syntax-entry ?\^m "> b"   table)

    table))

(defgroup rust-mode nil "Support for Rust code.")

(defcustom rust-indent-offset 4
  "*Indent Rust code by this number of spaces."
  :group 'rust-mode)

(defun rust-paren-level () (nth 0 (syntax-ppss)))
(defun rust-in-str-or-cmnt () (nth 8 (syntax-ppss)))
(defun rust-rewind-past-str-cmnt () (goto-char (nth 8 (syntax-ppss))))
(defun rust-rewind-irrelevant ()
  (let ((starting (point)))
    (skip-chars-backward "[:space:]\n")
    (if (looking-back "\\*/") (backward-char))
    (if (rust-in-str-or-cmnt)
        (rust-rewind-past-str-cmnt))
    (if (/= starting (point))
        (rust-rewind-irrelevant))))

(defun rust-align-to-expr-after-brace ()
  (save-excursion
    (forward-char)
    ;; We don't want to indent out to the open bracket if the
    ;; open bracket ends the line
    (when (not (looking-at "[[:blank:]]*\\(?://.*\\)?$"))
      (when (looking-at "[[:space:]]") (forward-to-word 1))
      (current-column))))

(defun rust-mode-indent-line ()
  (interactive)
  (let ((indent
         (save-excursion
           (back-to-indentation)
           (let ((level (rust-paren-level)))
             (cond
              ;; A function return type is indented to the corresponding function arguments
              ((looking-at "->")
               (save-excursion
                 (backward-list)
                 (or (rust-align-to-expr-after-brace)
                     (* rust-indent-offset (+ 1 level)))))

              ;; A closing brace is 1 level unindended
              ((looking-at "}") (* rust-indent-offset (- level 1)))

              ;; Doc comments in /** style with leading * indent to line up the *s
              ((and (nth 4 (syntax-ppss)) (looking-at "*"))
               (+ 1 (* rust-indent-offset level)))

              ;; If we're in any other token-tree / sexp, then:
              ;;  - [ or ( means line up with the opening token
              ;;  - { means indent to either nesting-level * rust-indent-offset,
              ;;    or one further indent from that if either current line
              ;;    begins with 'else', or previous line didn't end in
              ;;    semi, comma or brace (other than whitespace and line
              ;;    comments) , and wasn't an attribute.  But if we have 
              ;;    something after the open brace and ending with a comma,
              ;;    treat it as fields and align them.  PHEW.
              ((> level 0)
               (let ((pt (point)))
                 (rust-rewind-irrelevant)
                 (backward-up-list)
                 (or (and (looking-at "[[({]")
                          (rust-align-to-expr-after-brace))
                     (progn
                       (goto-char pt)
                       (back-to-indentation)
                       (if (looking-at "\\<else\\>")
                           (* rust-indent-offset (+ 1 level))
                         (progn
                           (goto-char pt)
                           (beginning-of-line)
                           (rust-rewind-irrelevant)
                           (end-of-line)
                           (if (looking-back
                                "[[,;{}(][[:space:]]*\\(?://.*\\)?")
                               (* rust-indent-offset level)
                             (back-to-indentation)
                             (if (looking-at "#")
                                 (* rust-indent-offset level)
                               (* rust-indent-offset (+ 1 level))))))))))

              ;; Otherwise we're in a column-zero definition
              (t 0))))))
    (cond
     ;; If we're to the left of the indentation, reindent and jump to it.
     ((<= (current-column) indent)
      (indent-line-to indent))

     ;; We're to the right; if it needs indent, do so but save excursion.
     ((not (eq (current-indentation) indent))
      (save-excursion (indent-line-to indent))))))


;; Font-locking definitions and helpers
(defconst rust-mode-keywords
  '("as"
    "break"
    "do"
    "else" "enum" "extern"
    "false" "fn" "for"
    "if" "impl" "in"
    "let" "loop"
    "match" "mod" "mut"
    "priv" "pub"
    "ref" "return"
    "self" "static" "struct" "super"
    "true" "trait" "type"
    "unsafe" "use"
    "while"))

(defconst rust-special-types
  '("u8" "i8"
    "u16" "i16"
    "u32" "i32"
    "u64" "i64"

    "f32" "f64"
    "float" "int" "uint"
    "bool"
    "str" "char"))

(defconst rust-re-ident "[[:word:][:multibyte:]_][[:word:][:multibyte:]_[:digit:]]*")
(defconst rust-re-CamelCase "[[:upper:]][[:word:][:multibyte:]_[:digit:]]*")
(defun rust-re-word (inner) (concat "\\<" inner "\\>"))
(defun rust-re-grab (inner) (concat "\\(" inner "\\)"))
(defun rust-re-grabword (inner) (rust-re-grab (rust-re-word inner)))
(defun rust-re-item-def (itype)
  (concat (rust-re-word itype) "[[:space:]]+" (rust-re-grab rust-re-ident)))

(defvar rust-mode-font-lock-keywords
  (append
   `(
     ;; Keywords proper
     (,(regexp-opt rust-mode-keywords 'words) . font-lock-keyword-face)

     ;; Special types
     (,(regexp-opt rust-special-types 'words) . font-lock-type-face)

     ;; Attributes like `#[bar(baz)]`
     (,(rust-re-grab (concat "#\\[" rust-re-ident "[^]]*\\]"))
      1 font-lock-preprocessor-face)

     ;; Syntax extension invocations like `foo!`, highlight including the !
     (,(concat (rust-re-grab (concat rust-re-ident "!")) "[({[:space:]]")
      1 font-lock-preprocessor-face)

     ;; Field names like `foo:`, highlight excluding the :
     (,(concat (rust-re-grab rust-re-ident) ":[^:]") 1 font-lock-variable-name-face)

     ;; Module names like `foo::`, highlight including the ::
     (,(rust-re-grab (concat rust-re-ident "::")) 1 font-lock-type-face)

     ;; Lifetimes like `'foo`
     (,(concat "'" (rust-re-grab rust-re-ident) "[^']") 1 font-lock-variable-name-face)

     ;; Character constants, since they're not treated as strings
     ;; in order to have sufficient leeway to parse 'lifetime above.
     (,(rust-re-grab "'[^']'") 1 font-lock-string-face)
     (,(rust-re-grab "'\\\\[nrt]'") 1 font-lock-string-face)
     (,(rust-re-grab "'\\\\x[[:xdigit:]]\\{2\\}'") 1 font-lock-string-face)
     (,(rust-re-grab "'\\\\u[[:xdigit:]]\\{4\\}'") 1 font-lock-string-face)
     (,(rust-re-grab "'\\\\U[[:xdigit:]]\\{8\\}'") 1 font-lock-string-face)

     ;; CamelCase Means Type Or Constructor
     (,(rust-re-grabword rust-re-CamelCase) 1 font-lock-type-face)
     )

   ;; Item definitions
   (loop for (item . face) in

         '(("enum" . font-lock-type-face)
           ("struct" . font-lock-type-face)
           ("type" . font-lock-type-face)
           ("mod" . font-lock-type-face)
           ("use" . font-lock-type-face)
           ("fn" . font-lock-function-name-face)
           ("static" . font-lock-constant-face))

         collect `(,(rust-re-item-def item) 1 ,face))))

(defun rust-fill-prefix-for-comment-start (line-start)
  "Determine what to use for `fill-prefix' based on what is at the beginning of a line."
  (let ((result 
         ;; Replace /* with same number of spaces
         (replace-regexp-in-string
          "\\(?:/\\*+\\)[!*]" 
          (lambda (s)
            ;; We want the * to line up with the first * of the comment start
            (concat (make-string (- (length s) 2) ?\x20) "*"))
          line-start)))
       ;; Make sure we've got at least one space at the end
    (if (not (= (aref result (- (length result) 1)) ?\x20))
        (setq result (concat result " ")))
    result))

(defun rust-in-comment-paragraph (body)
  ;; We might move the point to fill the next comment, but we don't want it
  ;; seeming to jump around on the user
  (save-excursion
    ;; If we're outside of a comment, with only whitespace and then a comment
    ;; in front, jump to the comment and prepare to fill it.
    (when (not (nth 4 (syntax-ppss)))
      (beginning-of-line)
      (when (looking-at (concat "[[:space:]\n]*" comment-start-skip))
        (goto-char (match-end 0))))

    ;; We need this when we're moving the point around and then checking syntax
    ;; while doing paragraph fills, because the cache it uses isn't always
    ;; invalidated during this.
    (syntax-ppss-flush-cache 1)
    ;; If we're at the beginning of a comment paragraph with nothing but
    ;; whitespace til the next line, jump to the next line so that we use the
    ;; existing prefix to figure out what the new prefix should be, rather than
    ;; inferring it from the comment start.
    (let ((next-bol (line-beginning-position 2)))
      (while (save-excursion
              (end-of-line)
              (syntax-ppss-flush-cache 1)
              (and (nth 4 (syntax-ppss))
                   (save-excursion 
                     (beginning-of-line)
                     (looking-at paragraph-start))
                   (looking-at "[[:space:]]*$")
                   (nth 4 (syntax-ppss next-bol))))
        (goto-char next-bol)))

    (syntax-ppss-flush-cache 1)
    ;; If we're on the last line of a multiline-style comment that started
    ;; above, back up one line so we don't mistake the * of the */ that ends
    ;; the comment for a prefix.
    (when (save-excursion
            (and (nth 4 (syntax-ppss (line-beginning-position 1)))
                 (looking-at "[[:space:]]*\\*/")))
      (goto-char (line-end-position 0)))
    (funcall body)))

(defun rust-with-comment-fill-prefix (body)
  (let*
      ((line-string (buffer-substring-no-properties 
                     (line-beginning-position) (line-end-position)))
       (line-comment-start
        (when (nth 4 (syntax-ppss)) 
          (cond
           ;; If we're inside the comment and see a * prefix, use it
           ((string-match "^\\([[:space:]]*\\*+[[:space:]]*\\)"
                          line-string)
            (match-string 1 line-string))
           ;; If we're at the start of a comment, figure out what prefix
           ;; to use for the subsequent lines after it
           ((string-match (concat "[[:space:]]*" comment-start-skip) line-string)
            (rust-fill-prefix-for-comment-start 
             (match-string 0 line-string))))))
       (fill-prefix 
        (or line-comment-start
            fill-prefix)))
    (funcall body)))

(defun rust-find-fill-prefix ()
  (rust-with-comment-fill-prefix (lambda () fill-prefix)))

(defun rust-fill-paragraph (&rest args)
  "Special wrapping for `fill-paragraph' to handle multi-line comments with a * prefix on each line."
  (rust-in-comment-paragraph
   (lambda () 
     (rust-with-comment-fill-prefix
      (lambda ()
        (let
            ((fill-paragraph-function
              (if (not (eq fill-paragraph-function 'rust-fill-paragraph))
                  fill-paragraph-function)))
          (apply 'fill-paragraph args)
          t))))))

(defun rust-do-auto-fill (&rest args)
  "Special wrapping for `do-auto-fill' to handle multi-line comments with a * prefix on each line."
  (rust-with-comment-fill-prefix
   (lambda ()
     (apply 'do-auto-fill args)
     t)))

(defun rust-fill-forward-paragraph (arg)
  ;; This is to work around some funny behavior when a paragraph separator is
  ;; at the very top of the file and there is a fill prefix.
  (let ((fill-prefix nil)) (forward-paragraph arg)))

(defun rust-comment-indent-new-line (&optional arg)
  (rust-with-comment-fill-prefix
   (lambda () (comment-indent-new-line arg))))

;; For compatibility with Emacs < 24, derive conditionally
(defalias 'rust-parent-mode
  (if (fboundp 'prog-mode) 'prog-mode 'fundamental-mode))


;;;###autoload
(define-derived-mode rust-mode rust-parent-mode "Rust"
  "Major mode for Rust code."
  :group 'rust-mode

  ;; Basic syntax
  (set-syntax-table rust-mode-syntax-table)

  ;; Indentation
  (set (make-local-variable 'indent-line-function)
       'rust-mode-indent-line)

  ;; Fonts
  (set (make-local-variable 'font-lock-defaults)
       '(rust-mode-font-lock-keywords nil nil nil nil))

  ;; Misc
  (set (make-local-variable 'comment-start) "// ")
  (set (make-local-variable 'comment-end)   "")
  (set (make-local-variable 'indent-tabs-mode) nil)

  ;; Allow paragraph fills for comments
  (set (make-local-variable 'comment-start-skip) 
       "\\(?://[/!]*\\|/\\*[*!]?\\)[[:space:]]*")
  (set (make-local-variable 'paragraph-start)
       (concat "[[:space:]]*\\(?:" comment-start-skip "\\|\\*/?[[:space:]]*\\|\\)$"))
  (set (make-local-variable 'paragraph-separate) paragraph-start)
  (set (make-local-variable 'normal-auto-fill-function) 'rust-do-auto-fill)
  (set (make-local-variable 'fill-paragraph-function) 'rust-fill-paragraph)
  (set (make-local-variable 'fill-forward-paragraph-function) 'rust-fill-forward-paragraph)
  (set (make-local-variable 'adaptive-fill-function) 'rust-find-fill-prefix)
  (set (make-local-variable 'comment-multi-line) t)
  (set (make-local-variable 'comment-line-break-function) 'rust-comment-indent-new-line)
  )


;;;###autoload
(add-to-list 'auto-mode-alist '("\\.rs$" . rust-mode))

(defun rust-mode-reload ()
  (interactive)
  (unload-feature 'rust-mode)
  (require 'rust-mode)
  (rust-mode))

(provide 'rust-mode)

;; Issue #6887: Rather than inheriting the 'gnu compilation error
;; regexp (which is broken on a few edge cases), add our own 'rust
;; compilation error regexp and use it instead.
(defvar rustc-compilation-regexps
  (let ((file "\\([^\n]+\\)")
        (start-line "\\([0-9]+\\)")
        (start-col  "\\([0-9]+\\)")
        (end-line   "\\([0-9]+\\)")
        (end-col    "\\([0-9]+\\)")
        (error-or-warning "\\(?:[Ee]rror\\|\\([Ww]arning\\)\\)"))
    (let ((re (concat "^" file ":" start-line ":" start-col
                      ": " end-line ":" end-col
                      " \\(?:[Ee]rror\\|\\([Ww]arning\\)\\):")))
      (cons re '(1 (2 . 4) (3 . 5) (6)))))
  "Specifications for matching errors in rustc invocations.
See `compilation-error-regexp-alist for help on their format.")

(eval-after-load 'compile
  '(progn
     (add-to-list 'compilation-error-regexp-alist-alist
                  (cons 'rustc rustc-compilation-regexps))
     (add-to-list 'compilation-error-regexp-alist 'rustc)))

;;; rust-mode.el ends here
