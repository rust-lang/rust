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

syn keyword   rustAssert      assert
syn keyword   rustKeyword     alt as break
syn keyword   rustKeyword     check claim cont const copy else export extern fail
syn keyword   rustKeyword     for if impl import in let log
syn keyword   rustKeyword     loop mod mut of pure
syn keyword   rustKeyword     ret self to unchecked
syn match     rustKeyword     "unsafe" " Allows also matching unsafe::foo()
syn keyword   rustKeyword     use while with
" FIXME: Scoped impl's name is also fallen in this category
syn keyword   rustKeyword     mod iface trait class enum type nextgroup=rustIdentifier skipwhite
syn keyword   rustKeyword     fn nextgroup=rustFuncName skipwhite

syn match     rustIdentifier  "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained
syn match     rustFuncName    "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained

" Reserved words
syn keyword   rustKeyword     m32 m64 m128 f80 f16 f128

syn keyword   rustType        any int uint float char bool u8 u16 u32 u64 f32
syn keyword   rustType        f64 i8 i16 i32 i64 str

syn keyword   rustBoolean     true false

syn keyword   rustConstant    some none       " option
" syn keyword   rustConstant    left right      " either
" syn keyword   rustConstant    ok err          " result
" syn keyword   rustConstant    success failure " task
" syn keyword   rustConstant    cons nil        " list
" syn keyword   rustConstant    empty node      " tree

" If foo::bar changes to foo.bar, change this ("::" to "\.").
" If foo::bar changes to Foo::bar, change this (first "\w" to "\u").
syn match     rustModPath     "\w\(\w\)*::"he=e-2,me=e-2
syn match     rustModPathSep  "::"

syn region    rustString      start=+L\="+ skip=+\\\\\|\\"+ end=+"+

syn region    rustAttribute   start="#\[" end="\]" contains=rustString

" Number literals
syn match     rustNumber      display "\<[0-9][0-9_]*\>"
syn match     rustNumber      display "\<[0-9][0-9_]*\(u\|u8\|u16\|u32\|u64\)\>"
syn match     rustNumber      display "\<[0-9][0-9_]*\(i8\|i16\|i32\|i64\)\>"

syn match     rustHexNumber   display "\<0x[a-fA-F0-9_]\+\>"
syn match     rustHexNumber   display "\<0x[a-fA-F0-9_]\+\(u\|u8\|u16\|u32\|u64\)\>"
syn match     rustHexNumber   display "\<0x[a-fA-F0-9_]\+\(i8\|i16\|i32\|i64\)\>"
syn match     rustBinNumber   display "\<0b[01_]\+\>"
syn match     rustBinNumber   display "\<0b[01_]\+\(u\|u8\|u16\|u32\|u64\)\>"
syn match     rustBinNumber   display "\<0b[01_]\+\(i8\|i16\|i32\|i64\)\>"

syn match     rustFloat       display "\<[0-9][0-9_]*\(f\|f32\|f64\)\>"
syn match     rustFloat       display "\<[0-9][0-9_]*\([eE][+-]\=[0-9_]\+\)\>"
syn match     rustFloat       display "\<[0-9][0-9_]*\([eE][+-]\=[0-9_]\+\)\(f\|f32\|f64\)\>"
syn match     rustFloat       display "\<[0-9][0-9_]*\.[0-9_]\+\>"
syn match     rustFloat       display "\<[0-9][0-9_]*\.[0-9_]\+\(f\|f32\|f64\)\>"
syn match     rustFloat       display "\<[0-9][0-9_]*\.[0-9_]\+\%([eE][+-]\=[0-9_]\+\)\>"
syn match     rustFloat       display "\<[0-9][0-9_]*\.[0-9_]\+\%([eE][+-]\=[0-9_]\+\)\(f\|f32\|f64\)\>"

syn match   rustCharacter   "'\([^'\\]\|\\\(['nrt\\\"]\|x\x\{2}\|u\x\{4}\|U\x\{8}\)\)'"

syn region    rustComment     start="/\*" end="\*/" contains=rustComment,rustTodo
syn region    rustComment     start="//" skip="\\$" end="$" contains=rustTodo keepend

syn keyword   rustTodo        TODO FIXME XXX NB

hi def link rustHexNumber     rustNumber
hi def link rustBinNumber     rustNumber

hi def link rustString        String
hi def link rustCharacter     Character
hi def link rustNumber        Number
hi def link rustBoolean       Boolean
hi def link rustConstant      Constant
hi def link rustFloat         Float
hi def link rustAssert        Keyword
hi def link rustKeyword       Keyword
hi def link rustIdentifier    Identifier
hi def link rustModPath       Include
hi def link rustFuncName      Function
hi def link rustComment       Comment
hi def link rustMacro         Macro
hi def link rustType          Type
hi def link rustTodo          Todo
hi def link rustAttribute     PreProc
" Other Suggestions:
" hi def link rustModPathSep    Conceal
" hi rustAssert ctermfg=yellow

syn sync minlines=200
syn sync maxlines=500

let b:current_syntax = "rust"
