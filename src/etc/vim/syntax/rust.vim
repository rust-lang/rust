" Vim syntax file
" Language:     Rust
" Maintainer:   Patrick Walton <pcwalton@mozilla.com>
" Last Change:  2010 Oct 13

" Quit when a syntax file was already loaded
if !exists("main_syntax")
  if version < 600
    syntax clear
  elseif exists("b:current_syntax")
    finish
  endif
  " we define it here so that included files can test for it
  let main_syntax='rust'
endif

syn keyword   rustKeyword     alt as assert auth be bind block break chan
syn keyword   rustKeyword     check claim cont const copy do else export fail
syn keyword   rustKeyword     fn for if import in inline lambda let log
syn keyword   rustKeyword     log_err mod mutable native note obj prove pure
syn keyword   rustKeyword     resource ret self tag type unsafe use while
syn keyword   rustKeyword     with

syn keyword   rustType        any int uint float char bool u8 u16 u32 u64 f32
syn keyword   rustType        f64 i8 i16 i32 i64 str task

syn keyword   rustBoolean     true false

syn match     rustItemPath    "\(\w\|::\)\+"

syn region	  rustString      start=+L\="+ skip=+\\\\\|\\"+ end=+"+

"integer number, or floating point number without a dot and with "f".
syn case ignore
syn match	  rustNumber		display contained "\d\+\(u\=l\{0,2}\|ll\=u\)\>"
"hex number
syn match	  rustNumber		display contained "0x\x\+\(u\=l\{0,2}\|ll\=u\)\>"
syn match	rustFloat		display contained "\d\+f"
"floating point number, with dot, optional exponent
syn match	rustFloat		display contained "\d\+\.\d*\(e[-+]\=\d\+\)\=[fl]\="
"floating point number, starting with a dot, optional exponent
syn match	rustFloat		display contained "\.\d\+\(e[-+]\=\d\+\)\=[fl]\=\>"
"floating point number, without dot, with exponent
syn match	rustFloat		display contained "\d\+e[-+]\=\d\+[fl]\=\>"

syn match   rustCharacter   "'[^']*'"

syn case match
syn region    rustComment     start="/\*" end="\*/"
syn region    rustComment     start="//" skip="\\$" end="$" keepend

hi def link rustString        String
hi def link rustCharacter     Character
hi def link rustNumber        Number
hi def link rustBoolean       Boolean
hi def link rustFloat         Float
hi def link rustKeyword       Keyword
hi def link rustComment       Comment
hi def link rustMacro         Macro
hi def link rustType          Type

let b:current_syntax = "rust"

