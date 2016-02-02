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
RUSTC = $(STAGE2_T_$(CFG_BUILD)_H_$(CFG_BUILD))
ifeq ($(CFG_OSTYPE),apple-darwin)
	FLEX_LDFLAGS=-ll
else
	FLEX_LDFLAGS=-lfl
endif

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

$(BG)RustLexer.class: $(BG) $(SG)RustLexer.g4
	$(Q)$(CFG_ANTLR4) -o $(BG) $(SG)RustLexer.g4
	$(Q)$(CFG_JAVAC) -d $(BG) $(BG)RustLexer.java

check-build-lexer-verifier: $(BG)verify

ifeq ($(NO_REBUILD),)
VERIFY_DEPS :=  rustc-stage2-H-$(CFG_BUILD) $(LD)stamp.rustc
else
VERIFY_DEPS :=
endif

$(BG)verify: $(BG) $(SG)verify.rs $(VERIFY_DEPS)
	$(Q)$(RUSTC) --out-dir $(BG) -L $(L) $(SG)verify.rs

ifdef CFG_JAVAC
ifdef CFG_ANTLR4
ifdef CFG_GRUN
check-lexer: $(BG) $(BG)RustLexer.class check-build-lexer-verifier
	$(info Verifying libsyntax against the reference lexer ...)
	$(Q)$(SG)check.sh $(S) "$(BG)" \
		"$(CFG_GRUN)" "$(BG)verify" "$(BG)RustLexer.tokens"
else
$(info cfg: lexer tooling not available, skipping lexer test...)
check-lexer:

endif
else
$(info cfg: lexer tooling not available, skipping lexer test...)
check-lexer:

endif
else
$(info cfg: lexer tooling not available, skipping lexer test...)
check-lexer:

endif

$(BG)lex.yy.c: $(SG)lexer.l $(BG)
	@$(call E, flex: $@)
	$(Q)$(CFG_FLEX) -o $@ $<

$(BG)lexer-lalr.o: $(BG)lex.yy.c $(BG)parser-lalr.tab.h
	@$(call E, cc: $@)
	$(Q)$(CFG_CC) -include $(BG)parser-lalr.tab.h -c -o $@ $<

$(BG)parser-lalr.tab.c $(BG)parser-lalr.tab.h: $(SG)parser-lalr.y
	@$(call E, bison: $@)
	$(Q)$(CFG_BISON) $< --output=$(BG)parser-lalr.tab.c --defines=$(BG)parser-lalr.tab.h \
		--name-prefix=rs --warnings=error=all

$(BG)parser-lalr.o: $(BG)parser-lalr.tab.c
	@$(call E, cc: $@)
	$(Q)$(CFG_CC) -c -o $@ $<

$(BG)parser-lalr-main.o: $(SG)parser-lalr-main.c
	@$(call E, cc: $@)
	$(Q)$(CFG_CC) -std=c99 -c -o $@ $<

$(BG)parser-lalr: $(BG)parser-lalr.o $(BG)parser-lalr-main.o $(BG)lexer-lalr.o
	@$(call E, cc: $@)
	$(Q)$(CFG_CC) -o $@ $^ $(FLEX_LDFLAGS)


ifdef CFG_FLEX
ifdef CFG_BISON
check-grammar: $(BG) $(BG)parser-lalr
	$(info Verifying grammar ...)
	$(SG)testparser.py -p $(BG)parser-lalr -s $(S)src

else
$(info cfg: bison not available, skipping parser test...)
check-grammar:

endif
else
$(info cfg: flex not available, skipping parser test...)
check-grammar:

endif
