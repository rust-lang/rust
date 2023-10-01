include ../tools.mk

# Test that rustdoc will properly load in a theme file and display it in the theme selector.

OUTPUT_DIR := "$(TMPDIR)/rustdoc-themes"

all:
	awk '/Begin theme: light/ {in_theme=1;next} /End theme:/ {in_theme=0} { if (in_theme) print }' \
		< '$(S)/src/librustdoc/html/static/css/noscript.css' > '$(TMPDIR)/test.css'
	$(RUSTDOC) -o $(OUTPUT_DIR) foo.rs --theme $(TMPDIR)/test.css
	$(HTMLDOCCK) $(OUTPUT_DIR) foo.rs
