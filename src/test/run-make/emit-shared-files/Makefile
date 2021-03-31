-include ../../run-make-fulldeps/tools.mk

CRATE_ONLY = $(TMPDIR)/crate-only
TOOLCHAIN_ONLY = $(TMPDIR)/toolchain-only
ALL_SHARED = $(TMPDIR)/all-shared

all: crate-only toolchain-only all-shared

crate-only:
	$(RUSTDOC) -Z unstable-options --emit=crate-specific --output $(CRATE_ONLY) --resource-suffix=-xxx --theme y.css x.rs
	[ -e $(CRATE_ONLY)/search-index-xxx.js ]
	[ -e $(CRATE_ONLY)/settings.html ]
	[ -e $(CRATE_ONLY)/x/all.html ]
	[ -e $(CRATE_ONLY)/x/index.html ]
	# FIXME: this probably shouldn't have a suffix
	[ -e $(CRATE_ONLY)/y-xxx.css ]
	! [ -e $(CRATE_ONLY)/storage-xxx.js ]
	! [ -e $(CRATE_ONLY)/SourceSerifPro-It.ttf.woff ]

toolchain-only:
	$(RUSTDOC) -Z unstable-options --emit=toolchain-shared-resources --output $(TOOLCHAIN_ONLY) --resource-suffix=-xxx x.rs
	[ -e $(TOOLCHAIN_ONLY)/storage-xxx.js ]
	! [ -e $(TOOLCHAIN_ONLY)/y-xxx.css ]
	! [ -e $(TOOLCHAIN_ONLY)/SourceSerifPro-It.ttf.woff ]
	! [ -e $(TOOLCHAIN_ONLY)/search-index-xxx.js ]
	! [ -e $(TOOLCHAIN_ONLY)/x/index.html ]

all-shared:
	$(RUSTDOC) -Z unstable-options --emit=toolchain-shared-resources,unversioned-shared-resources --output $(ALL_SHARED) --resource-suffix=-xxx x.rs
	[ -e $(ALL_SHARED)/storage-xxx.js ]
	[ -e $(ALL_SHARED)/SourceSerifPro-It.ttf.woff ]
	! [ -e $(ALL_SHARED)/y-xxx.css ]
	! [ -e $(ALL_SHARED)/search-index-xxx.js ]
	! [ -e $(ALL_SHARED)/settings.html ]
	! [ -e $(ALL_SHARED)/x ]
	! [ -e $(ALL_SHARED)/src ]
