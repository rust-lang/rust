-include ../tools.mk

tests: *.rs
	$(RUSTDOC) $< -o $(TMPDIR) --output-format json
	$(PYTHON) check_missing_items.py $(TMPDIR)/$(basename $<).json
	$(PYTHON) compare.py $(basename $<).expected $(TMPDIR)/$(basename $<).json
