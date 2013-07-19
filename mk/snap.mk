# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

define DEF_SNAP_FOR_STAGE_H
# $(1) stage
# $(2) triple

snap-stage$(1)-H-$(2): $$(HSREQ$(1)_H_$(2))
	$(CFG_PYTHON) $(S)src/etc/make-snapshot.py stage$(1) $(2)

endef

$(foreach host,$(CFG_HOST_TRIPLES),						\
 $(eval $(foreach stage,1 2 3,								\
  $(eval $(call DEF_SNAP_FOR_STAGE_H,$(stage),$(host))))))

snap-stage1: snap-stage1-H-$(CFG_BUILD_TRIPLE)

snap-stage2: snap-stage2-H-$(CFG_BUILD_TRIPLE)

snap-stage3: snap-stage3-H-$(CFG_BUILD_TRIPLE)
