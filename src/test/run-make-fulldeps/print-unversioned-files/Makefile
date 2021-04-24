-include ../tools.mk

all:
	$(RUSTDOC) -Z unstable-options --print unversioned-files | sort | diff - unversioned-files.txt
