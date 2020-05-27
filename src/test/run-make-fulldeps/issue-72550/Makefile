-include ../tools.mk


ifeq ($(UNAME),Darwin)
SNAPSHOT := apple.stderr
else
ifdef IS_WINDOWS
# TODO: Actually get the line numbers for windows and linux
# TODO: Find a way to prevent the snapshot from changing every time the
#       implementation files chage. Probably using `sed` or `cat-and-grep`?
SNAPSHOT := windows.stderr
else
# TODO: which other platforms use this branch?
SNAPSHOT := linux.stderr
endif
endif



all:
	rustc $(COMPILE_FLAGS) test.rs
	RUST_BACKTRACE=1 ./test 2> $(SNAPSHOT).actual && exit 1 || true
	# $(RUSTC) $(COMPILE_FLAGS) test.rs
	# $(call FAIL,test) > $(SNAPSHOT).actual
	[ -n "$(BLESS)" ] && echo "updating snapshot $(SNAPSHOT)" && mv $(SNAPSHOT).actual $(SNAPSHOT) \
		|| ( \
			diff -u $(SNAPSHOT).actual $(SNAPSHOT) && true rm $(SNAPSHOT).actual \
		) \
		|| ( \
			echo "stderr does not match snapshot $(SNAPSHOT)." && \
			echo "Re-run with BLESS=1 to update." && \
			exit 1 \
		)
