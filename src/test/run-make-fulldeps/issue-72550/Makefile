-include ../tools.mk


ifeq ($(UNAME),Darwin)
SNAPSHOT_NAME := apple.stderr
else
ifdef IS_WINDOWS
# TODO: Actually get the line numbers for windows and linux
# TODO: Find a way to prevent the snapshot from changing every time the
#       implementation files chage. Probably using `sed` or `cat-and-grep`?
SNAPSHOT_NAME := windows.stderr
else
# TODO: which other platforms use this branch?
SNAPSHOT_NAME := linux.stderr
endif
endif

ifdef BLESS
COMPARE_SNAPSHOTS := echo "updating snapshot $(SNAPSHOT_NAME)" && mv $(TMPDIR)/panic.stderr.actual $(SNAPSHOT_NAME)
else
COMPARE_SNAPSHOTS := diff -u $(SNAPSHOT_NAME) $(TMPDIR)/panic.stderr.actual \
	|| ( \
		echo "stderr does not match snapshot $(SNAPSHOT_NAME)." && \
		echo "Re-run with BLESS=1 to update." && \
		exit 1 \
	)
endif


all:
	$(RUSTC) -C debuginfo=2 test.rs
	RUST_BACKTRACE=1 $(call RUN,test) 2> $(TMPDIR)/panic.stderr.actual && exit 1 || exit 0
	$(COMPARE_SNAPSHOTS)
