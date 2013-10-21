# Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Create a way to reformat just some files
ifdef PPFILES
  PP_INPUTS_FILTERED := $(wildcard $(PPFILES))
else
  PP_INPUTS = $(wildcard $(addprefix $(S)src/libstd/,*.rs */*.rs)) \
              $(wildcard $(addprefix $(S)src/libextra/,*.rs */*.rs)) \
              $(wildcard $(addprefix $(S)src/rustc/,*.rs */*.rs */*/*.rs)) \
              $(wildcard $(S)src/test/*/*.rs    \
                         $(S)src/test/*/*/*.rs) \
              $(wildcard $(S)src/rustpkg/*.rs) \
              $(wildcard $(S)src/rust/*.rs)

  PP_INPUTS_FILTERED = $(shell echo $(PP_INPUTS) | xargs grep -L \
                       "no-reformat\|xfail-pretty\|xfail-test")
endif

reformat: $(SREQ1$(CFG_BUILD))
	@$(call E, reformat [stage1]: $@)
	for i in $(PP_INPUTS_FILTERED);  \
    do $(call CFG_RUN_TARG_$(CFG_BUILD),1,$(CFG_BUILD)/stage1/rustc$(X_$(CFG_BUILD))) \
       --pretty normal $$i >$$i.tmp; \
    if [ $$? -ne 0 ]; \
        then echo failed to print $$i; rm $$i.tmp; \
        else if cmp --silent $$i.tmp $$i; \
            then echo no changes to $$i; rm $$i.tmp; \
            else echo reformated $$i; mv $$i.tmp $$i; \
        fi; \
    fi; \
    done
