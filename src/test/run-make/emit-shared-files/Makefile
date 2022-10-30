include ../../run-make-fulldeps/tools.mk

INVOCATION_ONLY = $(TMPDIR)/invocation-only
TOOLCHAIN_ONLY = $(TMPDIR)/toolchain-only
ALL_SHARED = $(TMPDIR)/all-shared

all: invocation-only toolchain-only all-shared

invocation-only:
	$(RUSTDOC) -Z unstable-options --emit=invocation-specific --output $(INVOCATION_ONLY) --resource-suffix=-xxx --theme y.css --extend-css z.css x.rs
	[ -e $(INVOCATION_ONLY)/search-index-xxx.js ]
	[ -e $(INVOCATION_ONLY)/settings.html ]
	[ -e $(INVOCATION_ONLY)/x/all.html ]
	[ -e $(INVOCATION_ONLY)/x/index.html ]
	[ -e $(INVOCATION_ONLY)/theme-xxx.css ] # generated from z.css
	! [ -e $(INVOCATION_ONLY)/storage-xxx.js ]
	! [ -e $(INVOCATION_ONLY)/SourceSerif4-It.ttf.woff2 ]

	# FIXME: this probably shouldn't have a suffix
	[ -e $(INVOCATION_ONLY)/y-xxx.css ]
	# FIXME: this is technically incorrect (see `write_shared`)
	! [ -e $(INVOCATION_ONLY)/main-xxx.js ]

toolchain-only:
	$(RUSTDOC) -Z unstable-options --emit=toolchain-shared-resources --output $(TOOLCHAIN_ONLY) --resource-suffix=-xxx --extend-css z.css x.rs
	[ -e $(TOOLCHAIN_ONLY)/storage-xxx.js ]
	! [ -e $(TOOLCHAIN_ONLY)/SourceSerif4-It.ttf.woff2 ]
	! [ -e $(TOOLCHAIN_ONLY)/search-index-xxx.js ]
	! [ -e $(TOOLCHAIN_ONLY)/x/index.html ]
	! [ -e $(TOOLCHAIN_ONLY)/theme.css ]

	[ -e $(TOOLCHAIN_ONLY)/main-xxx.js ]
	! [ -e $(TOOLCHAIN_ONLY)/y-xxx.css ]

all-shared:
	$(RUSTDOC) -Z unstable-options --emit=toolchain-shared-resources,unversioned-shared-resources --output $(ALL_SHARED) --resource-suffix=-xxx --extend-css z.css x.rs
	[ -e $(ALL_SHARED)/storage-xxx.js ]
	[ -e $(ALL_SHARED)/SourceSerif4-It.ttf.woff2 ]
	! [ -e $(ALL_SHARED)/search-index-xxx.js ]
	! [ -e $(ALL_SHARED)/settings.html ]
	! [ -e $(ALL_SHARED)/x ]
	! [ -e $(ALL_SHARED)/src ]
	! [ -e $(ALL_SHARED)/theme.css ]

	[ -e $(ALL_SHARED)/main-xxx.js ]
	! [ -e $(ALL_SHARED)/y-xxx.css ]
