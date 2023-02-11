include ../tools.mk

all:
	# compile two different versions of crateA
	$(RUSTC) --crate-type=rlib crateA.rs -C metadata=-1 -C extra-filename=-1
	$(RUSTC) --crate-type=rlib crateA.rs -C metadata=-2 -C extra-filename=-2
	# make crateB depend on version 1 of crateA
	$(RUSTC) --crate-type=rlib crateB.rs --extern crateA=$(TMPDIR)/libcrateA-1.rlib
	# make crateC depend on version 2 of crateA
	$(RUSTC) crateC.rs --extern crateA=$(TMPDIR)/libcrateA-2.rlib 2>&1 | \
		tr -d '\r\n' | $(CGREP) -e \
	"mismatched types.*\
	crateB::try_foo\(foo2\);.*\
	expected \`crateA::foo::Foo\`, found \`Foo\`.*\
	different versions of crate \`crateA\`.*\
	mismatched types.*\
	crateB::try_bar\(bar2\);.*\
	expected trait \`crateA::bar::Bar\`, found trait \`Bar\`.*\
	different versions of crate \`crateA\`"
