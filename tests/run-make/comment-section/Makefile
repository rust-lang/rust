include ../tools.mk

# only-linux

all:
	echo 'fn main(){}' | $(RUSTC) - --emit=link,obj -Csave-temps --target=$(TARGET)

	# Check linked output has a `.comment` section with the expected content.
	readelf -p '.comment' $(TMPDIR)/rust_out | $(CGREP) -F 'rustc version 1.'

	# Check all object files (including temporary outputs) have a `.comment`
	# section with the expected content.
	set -e; for f in $(TMPDIR)/*.o; do \
		readelf -p '.comment' $$f | $(CGREP) -F 'rustc version 1.'; \
	done
