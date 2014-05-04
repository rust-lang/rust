-include ../tools.mk

all:
	$(RUSTC) write.rs
	$(CC) -o $(TMPDIR)/mknblock mknblock.c
	mkfifo $(TMPDIR)/fifo
	/bin/sh -c "(sleep 1; cat) < $(TMPDIR)/fifo > /dev/null" &
	/bin/sh -c "($(TMPDIR)/mknblock; $(TMPDIR)/write) > $(TMPDIR)/fifo"
