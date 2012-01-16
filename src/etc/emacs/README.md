rust-mode: A major emacs mode for editing Rust source code
==========================================================

`rust-mode` makes editing [Rust](http://rust-lang.org) code with emacs
enjoyable.

To install, check out this repository and add this to your .emacs
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
