# This runs the test for emacs rust-mode.
# It must be possible to find emacs via PATH.
emacs -batch -l rust-mode.el -l rust-mode-tests.el -f ert-run-tests-batch-and-exit
