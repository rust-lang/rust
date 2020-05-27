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
	$(RUSTC) $(COMPILE_FLAGS) test.rs
	# TODO: put $(SNAPSHOT).actual in a tmpdir somewhere.
	$(call FAIL,test) > $(SNAPSHOT).actual
	[ -n "$(BLESS)" ] && echo "updating snapshot $(SNAPSHOT)" && mv $(SNAPSHOT).actual $(SNAPSHOT) \
		|| ( \
			diff -u $(SNAPSHOT).actual $(SNAPSHOT) && true rm $(SNAPSHOT).actual \
		) \
		|| ( \
			echo "stderr does not match snapshot $(SNAPSHOT)." && \
			echo "Re-run with BLESS=1 to update." && \
			exit 1 \
		)
