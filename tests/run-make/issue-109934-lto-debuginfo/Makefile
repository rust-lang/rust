# ignore-cross-compile
include ../tools.mk

# With the upgrade to LLVM 16, this was getting:
#
#   error: Cannot represent a difference across sections
#
# The error stemmed from DI function definitions under type scopes, fixed by
# only declaring in type scope and defining the subprogram elsewhere.

all:
	$(RUSTC) lib.rs --test -C lto=fat -C debuginfo=2 -C incremental=$(TMPDIR)/inc-fat
