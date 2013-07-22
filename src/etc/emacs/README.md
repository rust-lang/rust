rust-mode: A major emacs mode for editing Rust source code
==========================================================

`rust-mode` makes editing [Rust](http://rust-lang.org) code with emacs
enjoyable.


### Manual Installation

To install manually, check out this repository and add this to your .emacs
file:

    (add-to-list 'load-path "/path/to/rust-mode/")
    (require 'rust-mode)

Make sure you byte-compile the .el files first, or the mode will be
painfully slow. There is an included `Makefile` which will do it for
you, so in the simplest case you can just run `make` and everything
should Just Work.

If for some reason that doesn't work, you can byte compile manually,
by pasting this in your `*scratch*` buffer, moving the cursor below
it, and pressing `C-j`:

    (progn
      (byte-compile-file "/path/to/rust-mode/cm-mode.el" t)
      (byte-compile-file "/path/to/rust-mode/rust-mode.el" t))

Rust mode will automatically be associated with .rs and .rc files. To
enable it explicitly, do `M-x rust-mode`.

### package.el installation via Marmalade or MELPA

It can be more convenient to use Emacs's package manager to handle
installation for you if you use many elisp libraries. If you have
package.el but haven't added Marmalade or MELPA, the community package source,
yet, add this to ~/.emacs.d/init.el:

Using Marmalade:

```lisp
(require 'package)
(add-to-list 'package-archives
             '("marmalade" . "http://marmalade-repo.org/packages/"))
(package-initialize)
```

Using MELPA:

```lisp
(require 'package)
(add-to-list 'package-archives
             '("melpa" . "http://melpa.milkbox.net/packages/") t)
(package-initialize)
```

Then do this to load the package listing:

* <kbd>M-x eval-buffer</kbd>
* <kbd>M-x package-refresh-contents</kbd>

If you use a version of Emacs prior to 24 that doesn't include
package.el, you can get it from http://bit.ly/pkg-el23.

If you have an older ELPA package.el installed from tromey.com, you
should upgrade in order to support installation from multiple sources.
The ELPA archive is deprecated and no longer accepting new packages,
so the version there (1.7.1) is very outdated.

#### Important

In order to have cm-mode properly initialized after compilation prior
to rust-mode.el compilation you will need to add these `advices` to
your init file or if you are a melpa user install the `melpa` package.

```lisp
(defadvice package-download-tar
  (after package-download-tar-initialize activate compile)
  "initialize the package after compilation"
  (package-initialize))

(defadvice package-download-single
  (after package-download-single-initialize activate compile)
  "initialize the package after compilation"
  (package-initialize))
```

#### Install rust-mode

From there you can install rust-mode or any other modes by choosing
them from a list:

* <kbd>M-x package-list-packages</kbd>

Now, to install packages, move your cursor to them and press i. This
will mark the packages for installation. When you're done with
marking, press x, and ELPA will install the packages for you (under
~/.emacs.d/elpa/).

* or using <kbd>M-x package-install rust-mode

### Known bugs

* Combining `global-whitespace-mode` and `rust-mode` is generally glitchy.
  See [Issue #3994](https://github.com/mozilla/rust/issues/3994).
