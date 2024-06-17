# needs-profiler-support
# ignore-cross-compile

include ../tools.mk

COMPILE_FLAGS=-g -Cprofile-generate="$(TMPDIR)"

all:
	$(RUSTC) $(COMPILE_FLAGS) test.rs
	$(call RUN,test) || exit 1
	[ -e "$(TMPDIR)"/default_*.profraw ] || (echo "No .profraw file"; exit 1)
