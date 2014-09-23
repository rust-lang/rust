;;; rust-mode-tests.el --- ERT tests for rust-mode.el

(require 'rust-mode)
(require 'ert)
(require 'cl)

(setq rust-test-fill-column 32)

(defun rust-compare-code-after-manip (original point-pos manip-func expected got)
  (equal expected got))

(defun rust-test-explain-bad-manip (original point-pos manip-func expected got)
  (if (equal expected got)
      nil
    (list
     ;; The (goto-char) and (insert) business here is just for
     ;; convenience--after an error, you can copy-paste that into emacs eval to
     ;; insert the bare strings into a buffer
     "Rust code was manipulated wrong after:"
     `(insert ,original)
     `(goto-char ,point-pos)
     'expected `(insert ,expected)
     'got `(insert ,got)
     (loop for i from 0 to (max (length original) (length expected))
           for oi = (if (< i (length got)) (elt got i))
           for ei = (if (< i (length expected)) (elt expected i))
           while (equal oi ei)
           finally return `(first-difference-at
                            (goto-char ,(+ 1 i))
                            expected ,(char-to-string ei)
                            got ,(char-to-string oi))))))
(put 'rust-compare-code-after-manip 'ert-explainer
     'rust-test-explain-bad-manip)

(defun rust-test-manip-code (original point-pos manip-func expected)
  (with-temp-buffer
    (rust-mode)
    (insert original)
    (goto-char point-pos)
    (funcall manip-func)
    (should (rust-compare-code-after-manip
             original point-pos manip-func expected (buffer-string)))))

(defun test-fill-paragraph (unfilled expected &optional start-pos end-pos)
  "We're going to run through many scenarios here--the point should be able to be anywhere from the start-pos (defaults to 1) through end-pos (defaults to the length of what was passed in) and (fill-paragraph) should return the same result.

Also, the result should be the same regardless of whether the code is at the beginning or end of the file.  (If you're not careful, that can make a difference.)  So we test each position given above with the passed code at the beginning, the end, neither and both.  So we do this a total of (end-pos - start-pos)*4 times.  Oy."
  (let* ((start-pos (or start-pos 1))
         (end-pos (or end-pos (length unfilled)))
         (padding "\n     \n")
         (padding-len (length padding)))
    (loop
     for pad-at-beginning from 0 to 1
     do (loop for pad-at-end from 0 to 1
              with padding-beginning = (if (= 0 pad-at-beginning) "" padding)
              with padding-end = (if (= 0 pad-at-end) "" padding)
              with padding-adjust = (* padding-len pad-at-beginning)
              with padding-beginning = (if (= 0 pad-at-beginning) "" padding)
              with padding-end = (if (= 0 pad-at-end) "" padding)
              ;; If we're adding space to the beginning, and our start position
              ;; is at the very beginning, we want to test within the added space.
              ;; Otherwise adjust the start and end for the beginning padding.
              with start-pos = (if (= 1 start-pos) 1 (+ padding-adjust start-pos))
              with end-pos = (+ end-pos padding-adjust)
              do (loop for pos from start-pos to end-pos
                       do (rust-test-manip-code
                           (concat padding-beginning unfilled padding-end)
                           pos
                           (lambda ()
                             (let ((fill-column rust-test-fill-column))
                               (fill-paragraph)))
                           (concat padding-beginning expected padding-end)))))))

(ert-deftest fill-paragraph-top-level-multi-line-style-doc-comment-second-line ()
  (test-fill-paragraph
   "/**
 * This is a very very very very very very very long string
 */"
   "/**
 * This is a very very very very
 * very very very long string
 */"))


(ert-deftest fill-paragraph-top-level-multi-line-style-doc-comment-first-line ()
  (test-fill-paragraph
   "/** This is a very very very very very very very long string
 */"
   "/** This is a very very very
 * very very very very long
 * string
 */"))

(ert-deftest fill-paragraph-multi-paragraph-multi-line-style-doc-comment ()
  (let
      ((multi-paragraph-unfilled
        "/**
 * This is the first really really really really really really really long paragraph
 *
 * This is the second really really really really really really long paragraph
 */"))
    (test-fill-paragraph
     multi-paragraph-unfilled
     "/**
 * This is the first really
 * really really really really
 * really really long paragraph
 *
 * This is the second really really really really really really long paragraph
 */"
     1 89)
    (test-fill-paragraph
     multi-paragraph-unfilled
     "/**
 * This is the first really really really really really really really long paragraph
 *
 * This is the second really
 * really really really really
 * really long paragraph
 */"
     90)))

(ert-deftest fill-paragraph-multi-paragraph-single-line-style-doc-comment ()
  (let
      ((multi-paragraph-unfilled
        "/// This is the first really really really really really really really long paragraph
///
/// This is the second really really really really really really long paragraph"))
    (test-fill-paragraph
     multi-paragraph-unfilled
     "/// This is the first really
/// really really really really
/// really really long paragraph
///
/// This is the second really really really really really really long paragraph"
     1 86)
    (test-fill-paragraph
     multi-paragraph-unfilled
     "/// This is the first really really really really really really really long paragraph
///
/// This is the second really
/// really really really really
/// really long paragraph"
     87)))

(ert-deftest fill-paragraph-multi-paragraph-single-line-style-indented ()
  (test-fill-paragraph
   "     // This is the first really really really really really really really long paragraph
     //
     // This is the second really really really really really really long paragraph"
   "     // This is the first really
     // really really really
     // really really really
     // long paragraph
     //
     // This is the second really really really really really really long paragraph" 1 89))

(ert-deftest fill-paragraph-multi-line-style-inner-doc-comment ()
  (test-fill-paragraph
   "/*! This is a very very very very very very very long string
 */"
   "/*! This is a very very very
 * very very very very long
 * string
 */"))

(ert-deftest fill-paragraph-single-line-style-inner-doc-comment ()
  (test-fill-paragraph
   "//! This is a very very very very very very very long string"
   "//! This is a very very very
//! very very very very long
//! string"))

(ert-deftest fill-paragraph-prefixless-multi-line-doc-comment ()
  (test-fill-paragraph
   "/**
This is my summary. Blah blah blah blah blah. Dilly dally dilly dally dilly dally doo.

This is some more text.  Fee fie fo fum.  Humpty dumpty sat on a wall.
*/"
   "/**
This is my summary. Blah blah
blah blah blah. Dilly dally
dilly dally dilly dally doo.

This is some more text.  Fee fie fo fum.  Humpty dumpty sat on a wall.
*/" 4 90))

(ert-deftest fill-paragraph-with-no-space-after-star-prefix ()
  (test-fill-paragraph
   "/**
 *This is a very very very very very very very long string
 */"
   "/**
 *This is a very very very very
 *very very very long string
 */"))

(ert-deftest fill-paragraph-single-line-style-with-code-before ()
  (test-fill-paragraph
   "fn foo() { }
/// This is my comment.  This is more of my comment.  This is even more."
   "fn foo() { }
/// This is my comment.  This is
/// more of my comment.  This is
/// even more." 14))

(ert-deftest fill-paragraph-single-line-style-with-code-after ()
  (test-fill-paragraph
   "/// This is my comment.  This is more of my comment.  This is even more.
fn foo() { }"
   "/// This is my comment.  This is
/// more of my comment.  This is
/// even more.
fn foo() { }" 1 73))

(ert-deftest fill-paragraph-single-line-style-code-before-and-after ()
  (test-fill-paragraph
   "fn foo() { }
/// This is my comment.  This is more of my comment.  This is even more.
fn bar() { }"
   "fn foo() { }
/// This is my comment.  This is
/// more of my comment.  This is
/// even more.
fn bar() { }" 14 67))

(defun test-auto-fill (initial position inserted expected)
  (rust-test-manip-code
   initial
   position
   (lambda ()
     (unwind-protect
         (progn
           (let ((fill-column rust-test-fill-column))
             (auto-fill-mode)
             (goto-char position)
             (insert inserted)
             (syntax-ppss-flush-cache 1)
             (funcall auto-fill-function)))
       (auto-fill-mode t)))
   expected))

(ert-deftest auto-fill-multi-line-doc-comment ()
  (test-auto-fill
   "/**
 *
 */"
   8
   "This is a very very very very very very very long string"
   "/**
 * This is a very very very very
 * very very very long string
 */"))

(ert-deftest auto-fill-single-line-doc-comment ()
  (test-auto-fill
   "/// This is the first really
/// really really really really
/// really really long paragraph
///
/// "
   103
   "This is the second really really really really really really long paragraph"
   "/// This is the first really
/// really really really really
/// really really long paragraph
///
/// This is the second really
/// really really really really
/// really long paragraph"
   ))

(ert-deftest auto-fill-multi-line-prefixless ()
  (test-auto-fill
   "/*

 */"
   4
   "This is a very very very very very very very long string"
   "/*
This is a very very very very
very very very long string
 */"
   ))

(defun test-indent (indented)
  (let ((deindented (replace-regexp-in-string "^[[:blank:]]*" "      " indented)))
    (rust-test-manip-code
     deindented
     1
     (lambda () (indent-region 1 (buffer-size)))
     indented)))


(ert-deftest indent-struct-fields-aligned ()
  (test-indent
   "
struct Foo { bar: int,
             baz: int }

struct Blah {x:int,
             y:int,
             z:String"))

(ert-deftest indent-doc-comments ()
  (test-indent
   "
/**
 * This is a doc comment
 *
 */

/// So is this

fn foo() {
    /*!
     * this is a nested doc comment
     */

    //! And so is this
}"))

(ert-deftest indent-inside-braces ()
  (test-indent
   "
// struct fields out one level:
struct foo {
    a:int,
    // comments too
    b:char
}

fn bar(x:Box<int>) {   // comment here should not affect the next indent
    bla();
    bla();
}"))

(ert-deftest indent-top-level ()
  (test-indent
   "
// Everything here is at the top level and should not be indented
#[attrib]
mod foo;

pub static bar = Quux{a: b()}

use foo::bar::baz;

fn foo() { }
"))

(ert-deftest indent-params-no-align ()
  (test-indent
   "
// Indent out one level because no params appear on the first line
fn xyzzy(
    a:int,
    b:char) { }

fn abcdef(
    a:int,
    b:char)
    -> char
{ }"))

(ert-deftest indent-params-align ()
  (test-indent
   "
// Align the second line of params to the first
fn foo(a:int,
       b:char) { }

fn bar(   a:int,
          b:char)
          -> int
{ }

fn baz(   a:int,  // shoudl work with a comment here
          b:char)
          -> int
{ }
"))

(ert-deftest indent-square-bracket-alignment ()
  (test-indent
   "
fn args_on_the_next_line( // with a comment
    a:int,
    b:String) {
    let aaaaaa = [
        1,
        2,
        3];
    let bbbbbbb = [1, 2, 3,
                   4, 5, 6];
    let ccc = [   10, 9, 8,
                  7, 6, 5];
}
"))

(ert-deftest indent-nested-fns ()
  (test-indent
   "
fn nexted_fns(a: fn(b:int,
                    c:char)
                    -> int,
              d: int)
              -> uint
{
    0
}
"
   ))

(ert-deftest indent-multi-line-expr ()
  (test-indent
   "
fn foo()
{
    x();
    let a =
        b();
}
"
   ))

(ert-deftest indent-match ()
  (test-indent
   "
fn foo() {
    match blah {
        Pattern => stuff(),
        _ => whatever
    }
}
"
   ))

(ert-deftest indent-match-multiline-pattern ()
  (test-indent
   "
fn foo() {
    match blah {
        Pattern |
        Pattern2 => {
            hello()
        },
        _ => whatever
    }
}
"
   ))

(ert-deftest indent-indented-match ()
  (test-indent
   "
fn foo() {
    let x =
        match blah {
            Pattern |
            Pattern2 => {
                hello()
            },
            _ => whatever
        };
    y();
}
"
   ))

(ert-deftest indent-curly-braces-within-parens ()
  (test-indent
   "
fn foo() {
    let x =
        foo(bar(|x| {
            only_one_indent_here();
        }));
    y();
}
"
   ))

(ert-deftest indent-weirdly-indented-block ()
  (rust-test-manip-code
   "
fn foo() {
 {
this_block_is_over_to_the_left_for_some_reason();
 }

}
"
   16
   #'indent-for-tab-command
   "
fn foo() {
 {
     this_block_is_over_to_the_left_for_some_reason();
 }

}
"
   ))

(ert-deftest indent-multi-line-attrib ()
  (test-indent
   "
#[attrib(
    this,
    that,
    theotherthing)]
fn function_with_multiline_attribute() {}
"
   ))


;; Make sure that in effort to cover match patterns we don't mistreat || or expressions
(ert-deftest indent-nonmatch-or-expression ()
  (test-indent
   "
fn foo() {
    let x = foo() ||
        bar();
}
"
   ))

(setq rust-test-motion-string
      "
fn fn1(arg: int) -> bool {
    let x = 5;
    let y = b();
    true
}

fn fn2(arg: int) -> bool {
    let x = 5;
    let y = b();
    true
}

pub fn fn3(arg: int) -> bool {
    let x = 5;
    let y = b();
    true
}

struct Foo {
    x: int
}
"
      rust-test-region-string rust-test-motion-string
      rust-test-indent-motion-string
      "
fn blank_line(arg:int) -> bool {

}

fn indenting_closing_brace() {
    if(true) {
}
}

fn indenting_middle_of_line() {
    if(true) {
 push_me_out();
} else {
               pull_me_back_in();
}
}

fn indented_already() {

    // The previous line already has its spaces
}
"

      ;; Symbol -> (line column)
      rust-test-positions-alist '((start-of-fn1 (2 0))
                                  (start-of-fn1-middle-of-line (2 15))
                                  (middle-of-fn1 (3 7))
                                  (end-of-fn1 (6 0))
                                  (between-fn1-fn2 (7 0))
                                  (start-of-fn2 (8 0))
                                  (middle-of-fn2 (10 4))
                                  (before-start-of-fn1 (1 0))
                                  (after-end-of-fn2 (13 0))
                                  (beginning-of-fn3 (14 0))
                                  (middle-of-fn3 (16 4))
                                  (middle-of-struct (21 10))
                                  (before-start-of-struct (19 0))
                                  (after-end-of-struct (23 0))
                                  (blank-line-indent-start (3 0))
                                  (blank-line-indent-target (3 4))
                                  (closing-brace-indent-start (8 1))
                                  (closing-brace-indent-target (8 5))
                                  (middle-push-indent-start (13 2))
                                  (middle-push-indent-target (13 9))
                                  (after-whitespace-indent-start (13 1))
                                  (after-whitespace-indent-target (13 8))
                                  (middle-pull-indent-start (15 19))
                                  (middle-pull-indent-target (15 12))
                                  (blank-line-indented-already-bol-start (20 0))
                                  (blank-line-indented-already-bol-target (20 4))
                                  (blank-line-indented-already-middle-start (20 2))
                                  (blank-line-indented-already-middle-target (20 4))
                                  (nonblank-line-indented-already-bol-start (21 0))
                                  (nonblank-line-indented-already-bol-target (21 4))
                                  (nonblank-line-indented-already-middle-start (21 2))
                                  (nonblank-line-indented-already-middle-target (21 4))))

(defun rust-get-buffer-pos (pos-symbol)
  "Get buffer position from POS-SYMBOL.

POS-SYMBOL is a symbol found in `rust-test-positions-alist'.
Convert the line-column information from that list into a buffer position value."
  (interactive "P")
  (pcase-let ((`(,line ,column) (cadr (assoc pos-symbol rust-test-positions-alist))))
    (save-excursion
      (goto-line line)
      (move-to-column column)
      (point))))

;;; FIXME: Maybe add an ERT explainer function (something that shows the
;;; surrounding code of the final point, not just the position).
(defun rust-test-motion (source-code init-pos final-pos manip-func &optional &rest args)
  "Test that MANIP-FUNC moves point from INIT-POS to FINAL-POS.

If ARGS are provided, send them to MANIP-FUNC.

INIT-POS, FINAL-POS are position symbols found in `rust-test-positions-alist'."
  (with-temp-buffer
    (rust-mode)
    (insert source-code)
    (goto-char (rust-get-buffer-pos init-pos))
    (apply manip-func args)
    (should (equal (point) (rust-get-buffer-pos final-pos)))))

(defun rust-test-region (source-code init-pos reg-beg reg-end manip-func &optional &rest args)
  "Test that MANIP-FUNC marks region from REG-BEG to REG-END.

INIT-POS is the initial position of point.
If ARGS are provided, send them to MANIP-FUNC.
All positions are position symbols found in `rust-test-positions-alist'."
  (with-temp-buffer
    (rust-mode)
    (insert source-code)
    (goto-char (rust-get-buffer-pos init-pos))
    (apply manip-func args)
    (should (equal (list (region-beginning) (region-end))
                   (list (rust-get-buffer-pos reg-beg)
                         (rust-get-buffer-pos reg-end))))))

(ert-deftest rust-beginning-of-defun-from-middle-of-fn ()
  (rust-test-motion
   rust-test-motion-string
   'middle-of-fn1
   'start-of-fn1
   #'beginning-of-defun))

(ert-deftest rust-beginning-of-defun-from-end ()
  (rust-test-motion
   rust-test-motion-string
   'end-of-fn1
   'start-of-fn1
   #'beginning-of-defun))

(ert-deftest rust-beginning-of-defun-before-open-brace ()
  (rust-test-motion
   rust-test-motion-string
   'start-of-fn1-middle-of-line
   'start-of-fn1
   #'beginning-of-defun))

(ert-deftest rust-beginning-of-defun-between-fns ()
  (rust-test-motion
   rust-test-motion-string
   'between-fn1-fn2
   'start-of-fn1
   #'beginning-of-defun))

(ert-deftest rust-beginning-of-defun-with-arg ()
  (rust-test-motion
   rust-test-motion-string
   'middle-of-fn2
   'start-of-fn1
   #'beginning-of-defun 2))

(ert-deftest rust-beginning-of-defun-with-negative-arg ()
  (rust-test-motion
   rust-test-motion-string
   'middle-of-fn1
   'beginning-of-fn3
   #'beginning-of-defun -2))

(ert-deftest rust-beginning-of-defun-pub-fn ()
  (rust-test-motion
   rust-test-motion-string
   'middle-of-fn3
   'beginning-of-fn3
   #'beginning-of-defun))

(ert-deftest rust-end-of-defun-from-middle-of-fn ()
  (rust-test-motion
   rust-test-motion-string
   'middle-of-fn1
   'between-fn1-fn2
   #'end-of-defun))

(ert-deftest rust-end-of-defun-from-beg ()
  (rust-test-motion
   rust-test-motion-string
   'start-of-fn1
   'between-fn1-fn2
   #'end-of-defun))

(ert-deftest rust-end-of-defun-before-open-brace ()
  (rust-test-motion
   rust-test-motion-string
   'start-of-fn1-middle-of-line
   'between-fn1-fn2
   #'end-of-defun))

(ert-deftest rust-end-of-defun-between-fns ()
  (rust-test-motion
   rust-test-motion-string
   'between-fn1-fn2
   'after-end-of-fn2
   #'end-of-defun))

(ert-deftest rust-end-of-defun-with-arg ()
  (rust-test-motion
   rust-test-motion-string
   'middle-of-fn1
   'after-end-of-fn2
   #'end-of-defun 2))

(ert-deftest rust-end-of-defun-with-negative-arg ()
  (rust-test-motion
   rust-test-motion-string
   'middle-of-fn3
   'between-fn1-fn2
   #'end-of-defun -2))

(ert-deftest rust-mark-defun-from-middle-of-fn ()
  (rust-test-region
   rust-test-region-string
   'middle-of-fn2
   'between-fn1-fn2 'after-end-of-fn2
   #'mark-defun))

(ert-deftest rust-mark-defun-from-end ()
  (rust-test-region
   rust-test-region-string
   'end-of-fn1
   'before-start-of-fn1 'between-fn1-fn2
   #'mark-defun))

(ert-deftest rust-mark-defun-start-of-defun ()
  (rust-test-region
   rust-test-region-string
   'start-of-fn2
   'between-fn1-fn2 'after-end-of-fn2
   #'mark-defun))

(ert-deftest rust-mark-defun-from-middle-of-struct ()
  (rust-test-region
   rust-test-region-string
   'middle-of-struct
   'before-start-of-struct 'after-end-of-struct
   #'mark-defun))

(ert-deftest indent-line-blank-line-motion ()
  (rust-test-motion
   rust-test-indent-motion-string
   'blank-line-indent-start
   'blank-line-indent-target
   #'indent-for-tab-command))

(ert-deftest indent-line-closing-brace-motion ()
  (rust-test-motion
   rust-test-indent-motion-string
   'closing-brace-indent-start
   'closing-brace-indent-target
   #'indent-for-tab-command))

(ert-deftest indent-line-middle-push-motion ()
  (rust-test-motion
   rust-test-indent-motion-string
   'middle-push-indent-start
   'middle-push-indent-target
   #'indent-for-tab-command))

(ert-deftest indent-line-after-whitespace-motion ()
  (rust-test-motion
   rust-test-indent-motion-string
   'after-whitespace-indent-start
   'after-whitespace-indent-target
   #'indent-for-tab-command))

(ert-deftest indent-line-middle-pull-motion ()
  (rust-test-motion
   rust-test-indent-motion-string
   'middle-pull-indent-start
   'middle-pull-indent-target
   #'indent-for-tab-command))

(ert-deftest indent-line-blank-line-indented-already-bol ()
  (rust-test-motion
   rust-test-indent-motion-string
   'blank-line-indented-already-bol-start
   'blank-line-indented-already-bol-target
   #'indent-for-tab-command))

(ert-deftest indent-line-blank-line-indented-already-middle ()
  (rust-test-motion
   rust-test-indent-motion-string
   'blank-line-indented-already-middle-start
   'blank-line-indented-already-middle-target
   #'indent-for-tab-command))

(ert-deftest indent-line-nonblank-line-indented-already-bol ()
  (rust-test-motion
   rust-test-indent-motion-string
   'nonblank-line-indented-already-bol-start
   'nonblank-line-indented-already-bol-target
   #'indent-for-tab-command))

(ert-deftest indent-line-nonblank-line-indented-already-middle ()
  (rust-test-motion
   rust-test-indent-motion-string
   'nonblank-line-indented-already-middle-start
   'nonblank-line-indented-already-middle-target
   #'indent-for-tab-command))

(defun rust-test-fontify-string (str)
  (with-temp-buffer
    (rust-mode)
    (insert str)
    (font-lock-fontify-buffer)
    (buffer-string)))

(defun rust-test-group-str-by-face (str)
  "Fontify `STR' in rust-mode and group it by face, returning a
list of substrings of `STR' each followed by its face."
  (cl-loop with fontified = (rust-test-fontify-string str)
           for start = 0 then end
           while start
           for end   = (next-single-property-change start 'face fontified)
           for prop  = (get-text-property start 'face fontified)
           for text  = (substring-no-properties fontified start end)
           if prop
           append (list text prop)))

(defun rust-test-font-lock (source face-groups)
  "Test that `SOURCE' fontifies to the expected `FACE-GROUPS'"
  (should (equal (rust-test-group-str-by-face source)
                 face-groups)))

(ert-deftest font-lock-attribute-simple ()
  (rust-test-font-lock
   "#[foo]"
   '("#[foo]" font-lock-preprocessor-face)))

(ert-deftest font-lock-attribute-inner ()
  (rust-test-font-lock
   "#![foo]"
   '("#![foo]" font-lock-preprocessor-face)))

(ert-deftest font-lock-attribute-key-value ()
  (rust-test-font-lock
   "#[foo = \"bar\"]"
   '("#[foo = " font-lock-preprocessor-face
     "\"bar\"" font-lock-string-face
     "]" font-lock-preprocessor-face)))

(ert-deftest font-lock-attribute-around-comment ()
  (rust-test-font-lock
   "#[foo /* bar */]"
   '("#[foo " font-lock-preprocessor-face
     "/* " font-lock-comment-delimiter-face
     "bar */" font-lock-comment-face
     "]" font-lock-preprocessor-face)))

(ert-deftest font-lock-attribute-inside-string ()
  (rust-test-font-lock
   "\"#[foo]\""
   '("\"#[foo]\"" font-lock-string-face)))

(ert-deftest font-lock-attribute-inside-comment ()
  (rust-test-font-lock
   "/* #[foo] */"
   '("/* " font-lock-comment-delimiter-face
     "#[foo] */" font-lock-comment-face)))
