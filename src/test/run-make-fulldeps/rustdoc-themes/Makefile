include ../tools.mk

# Test that rustdoc will properly load in a theme file and display it in the theme selector.

OUTPUT_DIR := "$(TMPDIR)/rustdoc-themes"

all:
	cp $(S)/src/librustdoc/html/static/css/themes/light.css $(TMPDIR)/test.css
	$(RUSTDOC) -o $(OUTPUT_DIR) foo.rs --theme $(TMPDIR)/test.css
	$(HTMLDOCCK) $(OUTPUT_DIR) foo.rs
