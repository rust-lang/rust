E=@echo
TEMP=temp.el

EMACS ?= emacs

all: $(TEMP)
	$(EMACS) -batch -q -no-site-file -l ./$(TEMP) -f rustmode-compile
	rm -f $(TEMP)
$(TEMP):
	$(E) '(setq load-path (cons "." load-path))' >> $(TEMP)
	$(E) '(defun rustmode-compile () (mapcar (lambda (x) (byte-compile-file x))' >> $(TEMP)
	$(E) ' (list "cm-mode.el" "rust-mode.el")))' >> $(TEMP)
clean:
	rm -f *.elc $(TEMP)
