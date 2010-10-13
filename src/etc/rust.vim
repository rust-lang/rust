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

syn keyword   rustKeyword     use meta syntax mutable native mod import export
syn keyword   rustKeyword     let auto io state unsafe auth with bind type true
syn keyword   rustKeyword     false any int uint float char bool u8 u16 u32 u64
syn keyword   rustKeyword     f32 i8 i16 i32 i64 f64 rec tup tag vec str fn
syn keyword   rustKeyword     iter obj as drop task port chan flush spawn if
syn keyword   rustKeyword     else alt case in do while break cont fail log
syn keyword   rustKeyword     note claim check prove for each ret put be

syn region	  rustString		  start=+L\="+ skip=+\\\\\|\\"+ end=+"+

syn region    rustComment     start="/\*" end="\*/"
syn region    rustComment     start="//" skip="\\$" end="$" keepend

hi def link rustString        String
hi def link rustKeyword       Keyword
hi def link rustComment       Comment

let b:current_syntax = "rust"

