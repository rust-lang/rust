# only-linux
# ignore-arm - linker error on `armhf-gnu`

include ../tools.mk

# Issue #66530: We would ICE if someone compiled with `-o /dev/null`,
# because we would try to generate auxiliary files in `/dev/` (which
# at least the OS X file system rejects).
#
# An attempt to `-Ztemps-dir` into a directory we cannot write into should
# indeed be an error; but not an ICE.
#
# However, some folks run tests as root, which can write `/dev/` and end
# up clobbering `/dev/null`. Instead we'll use an inaccessible path, which
# also used to ICE, but even root can't magically write there.
#
# Note that `-Ztemps-dir` uses `create_dir_all` so it is not sufficient to
# use a directory with non-existing parent like `/does-not-exist/output`.

all:
	# Create an inaccessible directory
	mkdir $(TMPDIR)/inaccessible
	chmod 000 $(TMPDIR)/inaccessible

	# Run rustc with `-Ztemps-dir` set to a directory 
	# *inside* the inaccessible one, so that it can't create it
	$(RUSTC) program.rs -Ztemps-dir=$(TMPDIR)/inaccessible/tmp 2>&1 \
		| $(CGREP) 'failed to find or create the directory specified by `--temps-dir`'

	# Make the inaccessible directory accessible,
	# so that compiletest can delete the temp dir
	chmod +rw $(TMPDIR)/inaccessible
