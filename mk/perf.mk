# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.


ifdef CFG_PERF_TOOL
rustc-perf$(X): $(CFG_BUILD)/stage2/bin/rustc$(X_$(CFG_BUILD))
	@$(call E, perf compile: $@)
	$(PERF_STAGE2_T_$(CFG_BUILD)_H_$(CFG_BUILD)) \
		 -o $@ $(COMPILER_CRATE) >rustc-perf.err 2>&1
	$(Q)rm -f $(LIBRUSTC_GLOB)
else
rustc-perf$(X): $(CFG_BUILD)/stage2/bin/rustc$(X_$(CFG_BUILD))
	$(Q)touch $@
endif

perf: check-stage2-perf rustc-perf$(X_$(CFG_BUILD))
	$(Q)find $(CFG_BUILD)/test/perf -name \*.err | xargs cat
	$(Q)cat rustc-perf.err
