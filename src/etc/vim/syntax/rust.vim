" Vim syntax file
" Language:     Rust
" Maintainer:   Patrick Walton <pcwalton@mozilla.com>
" Maintainer:   Ben Blum <bblum@mozilla.com>
" Last Change:  2012 Jul 06

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn keyword   rustAssert      assert
syn match     rustAssert      "assert\(\w\)*"
syn keyword   rustKeyword     alt again as break
syn keyword   rustKeyword     check claim const copy do drop else export extern fail
syn keyword   rustKeyword     for if impl import in let log
syn keyword   rustKeyword     loop mod mut new of owned pure
syn keyword   rustKeyword     ret self to unchecked
syn match     rustKeyword     "unsafe" " Allows also matching unsafe::foo()
syn keyword   rustKeyword     use while with
" FIXME: Scoped impl's name is also fallen in this category
syn keyword   rustKeyword     mod trait class struct enum type nextgroup=rustIdentifier skipwhite
syn keyword   rustKeyword     fn nextgroup=rustFuncName skipwhite

syn match     rustIdentifier  contains=rustIdentifierPrime "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained
syn match     rustFuncName    "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained

" Reserved words
syn keyword   rustKeyword     m32 m64 m128 f80 f16 f128

syn keyword   rustType        any int uint float char bool u8 u16 u32 u64 f32
syn keyword   rustType        f64 i8 i16 i32 i64 str
syn keyword   rustType        option either

" Types from libc
syn keyword   rustType        c_float c_double c_void FILE fpos_t
syn keyword   rustType        DIR dirent
syn keyword   rustType        c_char c_schar c_uchar
syn keyword   rustType        c_short c_ushort c_int c_uint c_long c_ulong
syn keyword   rustType        size_t ptrdiff_t clock_t time_t
syn keyword   rustType        c_longlong c_ulonglong intptr_t uintptr_t
syn keyword   rustType        off_t dev_t ino_t pid_t mode_t ssize_t

syn keyword   rustBoolean     true false

syn keyword   rustConstant    some none       " option
syn keyword   rustConstant    left right      " either
syn keyword   rustConstant    ok err          " result
syn keyword   rustConstant    success failure " task
syn keyword   rustConstant    cons nil        " list
" syn keyword   rustConstant    empty node      " tree

" Constants from libc
syn keyword   rustConstant    EXIT_FAILURE EXIT_SUCCESS RAND_MAX
syn keyword   rustConstant    EOF SEEK_SET SEEK_CUR SEEK_END _IOFBF _IONBF
syn keyword   rustConstant    _IOLBF BUFSIZ FOPEN_MAX FILENAME_MAX L_tmpnam
syn keyword   rustConstant    TMP_MAX O_RDONLY O_WRONLY O_RDWR O_APPEND O_CREAT
syn keyword   rustConstant    O_EXCL O_TRUNC S_IFIFO S_IFCHR S_IFBLK S_IFDIR
syn keyword   rustConstant    S_IFREG S_IFMT S_IEXEC S_IWRITE S_IREAD S_IRWXU
syn keyword   rustConstant    S_IXUSR S_IWUSR S_IRUSR F_OK R_OK W_OK X_OK
syn keyword   rustConstant    STDIN_FILENO STDOUT_FILENO STDERR_FILENO

" If foo::bar changes to foo.bar, change this ("::" to "\.").
" If foo::bar changes to Foo::bar, change this (first "\w" to "\u").
syn match     rustModPath     "\w\(\w\)*::[^<]"he=e-3,me=e-3
syn match     rustModPathSep  "::"

syn match     rustFuncCall    "\w\(\w\)*("he=e-1,me=e-1 contains=rustAssert
syn match     rustFuncCall    "\w\(\w\)*::<"he=e-3,me=e-3 contains=rustAssert " foo::<T>();

syn match     rustMacro       '\w\(\w\)*!'
syn match     rustMacro       '#\w\(\w\)*'

syn region    rustString      start=+L\="+ skip=+\\\\\|\\"+ end=+"+ contains=rustTodo

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

" For those who don't want to see `::`...
syn match   rustModPathSep  "::" conceal cchar=ㆍ

syn match rustArrowHead contained ">" conceal cchar= 
syn match rustArrowTail contained "-" conceal cchar=⟶
syn match rustArrowFull "->" contains=rustArrowHead,rustArrowTail

syn match rustFatArrowHead contained ">" conceal cchar= 
syn match rustFatArrowTail contained "=" conceal cchar=⟹
syn match rustFatArrowFull "=>" contains=rustFatArrowHead,rustFatArrowTail

syn match rustIdentifierPrime /\<\@!_\(_*\>\)\@=/ conceal cchar=′

hi def link rustHexNumber       rustNumber
hi def link rustBinNumber       rustNumber
hi def link rustIdentifierPrime rustIdentifier

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
hi def link rustModPathSep    Conceal
" Other Suggestions:
" hi rustAssert ctermfg=yellow
" hi rustMacro ctermfg=magenta

syn sync minlines=200
syn sync maxlines=500

let b:current_syntax = "rust"
