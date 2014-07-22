# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

BG = $(CFG_BUILD_DIR)/grammar/
SG = $(S)src/grammar/
B = $(CFG_BUILD_DIR)/$(CFG_BUILD)/stage2/
L = $(B)lib/rustlib/$(CFG_BUILD)/lib
LD = $(CFG_BUILD)/stage2/lib/rustlib/$(CFG_BUILD)/lib/
RUSTC = $(B)bin/rustc

# Run the reference lexer against libsyntax and compare the tokens and spans.
# If "// ignore-lexer-test" is present in the file, it will be ignored.
#
# $(1) is the file to test.
define LEXER_TEST
grep "// ignore-lexer-test" $(1) ; \
  if [ $$? -eq 1 ]; then \
   CLASSPATH=$(B)grammar $(CFG_GRUN) RustLexer tokens -tokens < $(1) \
   | $(B)grammar/verify $(1) ; \
  fi
endef

$(BG):
	$(Q)mkdir -p $(BG)

$(BG)RustLexer.class: $(SG)RustLexer.g4
	$(Q)$(CFG_ANTLR4) -o $(B)grammar $(SG)RustLexer.g4
	$(Q)$(CFG_JAVAC) -d $(BG) $(BG)RustLexer.java

$(BG)verify: $(SG)verify.rs rustc-stage2-H-$(CFG_BUILD) $(LD)stamp.regex_macros $(LD)stamp.rustc
	$(Q)$(RUSTC) -O --out-dir $(BG) -L $(L) $(SG)verify.rs

check-lexer: $(BG) $(BG)RustLexer.class $(BG)verify
ifdef CFG_JAVAC
ifdef CFG_ANTLR4
ifdef CFG_GRUN
	$(info Verifying libsyntax against the reference lexer ...)
	$(Q)$(SG)check.sh $(S) "$(BG)" \
		"$(CFG_GRUN)" "$(BG)verify" "$(BG)RustLexer.tokens"
else
$(info grun not available, skipping lexer test...)
endif
else
$(info antlr4 not available, skipping lexer test...)
endif
else
$(info javac not available, skipping lexer test...)
endif
