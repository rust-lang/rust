" Vim syntax file
" Language:     Rust
" Maintainer:   Patrick Walton <pcwalton@mozilla.com>
" Maintainer:   Ben Blum <bblum@cs.cmu.edu>
" Maintainer:   Chris Morgan <me@chrismorgan.info>
" Last Change:  2013 Sep 4

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

" Syntax definitions {{{1
" Basic keywords {{{2
syn keyword   rustConditional match if else
syn keyword   rustOperator    as

syn match     rustAssert      "\<assert\(\w\)*!" contained
syn match     rustFail        "\<fail\(\w\)*!" contained
syn keyword   rustKeyword     break do extern
syn keyword   rustKeyword     in if impl let log
syn keyword   rustKeyword     for impl let log
syn keyword   rustKeyword     loop mod once priv pub
syn keyword   rustKeyword     return
syn keyword   rustKeyword     unsafe while
syn keyword   rustKeyword     use nextgroup=rustModPath skipwhite
" FIXME: Scoped impl's name is also fallen in this category
syn keyword   rustKeyword     mod trait struct enum type nextgroup=rustIdentifier skipwhite
syn keyword   rustKeyword     fn nextgroup=rustFuncName skipwhite
syn keyword   rustStorage     const mut ref static

syn match     rustIdentifier  contains=rustIdentifierPrime "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained
syn match     rustFuncName    "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained

" Reserved (but not yet used) keywords {{{2
syn keyword   rustKeyword     be yield typeof

" Built-in types {{{2
syn keyword   rustType        int uint float char bool u8 u16 u32 u64 f32
syn keyword   rustType        f64 i8 i16 i32 i64 str Self

" Things from the prelude (src/libstd/prelude.rs) {{{2
" This section is just straight transformation of the contents of the prelude,
" to make it easy to update.

" Core operators {{{3
syn keyword   rustEnum        Either
syn keyword   rustEnumVariant Left Right
syn keyword   rustTrait       Sized
syn keyword   rustTrait       Freeze Send
syn keyword   rustTrait       Add Sub Mul Div Rem Neg Not
syn keyword   rustTrait       BitAnd BitOr BitXor
syn keyword   rustTrait       Drop
syn keyword   rustTrait       Shl Shr Index
syn keyword   rustEnum        Option
syn keyword   rustEnumVariant Some None
syn keyword   rustEnum        Result
syn keyword   rustEnumVariant Ok Err

" Functions {{{3
"syn keyword rustFunction print println
"syn keyword rustFunction range

" Types and traits {{{3
syn keyword rustTrait ToCStr
syn keyword rustTrait Clone DeepClone
syn keyword rustTrait Eq ApproxEq Ord TotalEq TotalOrd Ordering Equiv
syn keyword rustEnumVariant Less Equal Greater
syn keyword rustTrait Char
syn keyword rustTrait Container Mutable Map MutableMap Set MutableSet
syn keyword rustTrait Hash
syn keyword rustTrait Times
syn keyword rustTrait FromIterator Extendable
syn keyword rustTrait Iterator DoubleEndedIterator RandomAccessIterator ClonableIterator
syn keyword rustTrait OrdIterator MutableDoubleEndedIterator ExactSize
syn keyword rustTrait Num NumCast CheckedAdd CheckedSub CheckedMul
syn keyword rustTrait Orderable Signed Unsigned Round
syn keyword rustTrait Algebraic Trigonometric Exponential Hyperbolic
syn keyword rustTrait Integer Fractional Real RealExt
syn keyword rustTrait Bitwise BitCount Bounded
syn keyword rustTrait Primitive Int Float ToStrRadix
syn keyword rustTrait GenericPath
syn keyword rustTrait Path
syn keyword rustTrait PosixPath
syn keyword rustTrait WindowsPath
syn keyword rustTrait RawPtr
syn keyword rustTrait Ascii AsciiCast OwnedAsciiCast AsciiStr ToBytesConsume
syn keyword rustTrait Str StrVector StrSlice OwnedStr
syn keyword rustTrait FromStr
syn keyword rustTrait IterBytes
syn keyword rustTrait ToStr ToStrConsume
syn keyword rustTrait CopyableTuple ImmutableTuple
syn keyword rustTrait CloneableTuple1 ImmutableTuple1
syn keyword rustTrait CloneableTuple2 CloneableTuple3 CloneableTuple4 CloneableTuple5
syn keyword rustTrait CloneableTuple6 CloneableTuple7 CloneableTuple8 CloneableTuple9
syn keyword rustTrait CloneableTuple10 CloneableTuple11 CloneableTuple12
syn keyword rustTrait ImmutableTuple2 ImmutableTuple3 ImmutableTuple4 ImmutableTuple5
syn keyword rustTrait ImmutableTuple6 ImmutableTuple7 ImmutableTuple8 ImmutableTuple9
syn keyword rustTrait ImmutableTuple10 ImmutableTuple11 ImmutableTuple12
syn keyword rustTrait Vector VectorVector CopyableVector ImmutableVector
syn keyword rustTrait ImmutableEqVector ImmutableTotalOrdVector ImmutableCopyableVector
syn keyword rustTrait OwnedVector OwnedCopyableVector OwnedEqVector MutableVector
syn keyword rustTrait Reader ReaderUtil Writer WriterUtil
syn keyword rustTrait Default

"syn keyword rustFunction stream
syn keyword rustTrait Port Chan GenericChan GenericSmartChan GenericPort Peekable
"syn keyword rustFunction spawn

syn keyword   rustSelf        self
syn keyword   rustBoolean     true false

syn keyword   rustConstant    Some None       " option
syn keyword   rustConstant    Left Right      " either
syn keyword   rustConstant    Ok Err          " result
syn keyword   rustConstant    Less Equal Greater " Ordering

" Other syntax {{{2

" If foo::bar changes to foo.bar, change this ("::" to "\.").
" If foo::bar changes to Foo::bar, change this (first "\w" to "\u").
syn match     rustModPath     "\w\(\w\)*::[^<]"he=e-3,me=e-3
syn match     rustModPath     "\w\(\w\)*" contained " only for 'use path;'
syn match     rustModPathSep  "::"

syn match     rustFuncCall    "\w\(\w\)*("he=e-1,me=e-1
syn match     rustFuncCall    "\w\(\w\)*::<"he=e-3,me=e-3 " foo::<T>();

" This is merely a convention; note also the use of [A-Z], restricting it to
" latin identifiers rather than the full Unicode uppercase. I have not used
" [:upper:] as it depends upon 'noignorecase'
"syn match     rustCapsIdent    display "[A-Z]\w\(\w\)*"

syn match     rustOperator     display "\%(+\|-\|/\|*\|=\|\^\|&\||\|!\|>\|<\|%\)=\?"
" This one isn't *quite* right, as we could have binary-& with a reference
syn match     rustSigil        display /&\s\+[&~@*][^)= \t\r\n]/he=e-1,me=e-1
syn match     rustSigil        display /[&~@*][^)= \t\r\n]/he=e-1,me=e-1
" This isn't actually correct; a closure with no arguments can be `|| { }`.
" Last, because the & in && isn't a sigil
syn match     rustOperator     display "&&\|||"

syn match     rustMacro       '\w\(\w\)*!' contains=rustAssert,rustFail
syn match     rustMacro       '#\w\(\w\)*' contains=rustAssert,rustFail

syn match     rustFormat      display "%\(\d\+\$\)\=[-+' #0*]*\(\d*\|\*\|\*\d\+\$\)\(\.\(\d*\|\*\|\*\d\+\$\)\)\=\([hlLjzt]\|ll\|hh\)\=\([aAbdiuoxXDOUfFeEgGcCsSpn?]\|\[\^\=.[^]]*\]\)" contained
syn match     rustFormat      display "%%" contained
syn match     rustSpecial     display contained /\\\([nrt\\'"]\|x\x\{2}\|u\x\{4}\|U\x\{8}\)/
syn match     rustStringContinuation display contained /\\\n\s*/
syn region    rustString      start=+"+ skip=+\\\\\|\\"+ end=+"+ contains=rustTodo,rustFormat,rustSpecial,rustStringContinuation

syn region    rustAttribute   start="#\[" end="\]" contains=rustString,rustDeriving
syn region    rustDeriving    start="deriving(" end=")" contained contains=rustTrait

" Number literals
syn match     rustNumber      display "\<[0-9][0-9_]*\>"
syn match     rustNumber      display "\<[0-9][0-9_]*\(u\|u8\|u16\|u32\|u64\)\>"
syn match     rustNumber      display "\<[0-9][0-9_]*\(i\|i8\|i16\|i32\|i64\)\>"

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

" For the benefit of delimitMate
syn region rustLifetimeCandidate display start=/&'\%(\([^'\\]\|\\\(['nrt\\\"]\|x\x\{2}\|u\x\{4}\|U\x\{8}\)\)'\)\@!/ end=/[[:cntrl:][:space:][:punct:]]\@=\|$/ contains=rustSigil,rustLifetime
syn region rustGenericRegion display start=/<\%('\|[^[cntrl:][:space:][:punct:]]\)\@=')\S\@=/ end=/>/ contains=rustGenericLifetimeCandidate
syn region rustGenericLifetimeCandidate display start=/\%(<\|,\s*\)\@<='/ end=/[[:cntrl:][:space:][:punct:]]\@=\|$/ contains=rustSigil,rustLifetime

"rustLifetime must appear before rustCharacter, or chars will get the lifetime highlighting
syn match     rustLifetime    display "\'\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*"
syn match   rustCharacter   /'\([^'\\]\|\\\([nrt\\'"]\|x\x\{2}\|u\x\{4}\|U\x\{8}\)\)'/ contains=rustSpecial

syn region    rustCommentML   start="/\*" end="\*/" contains=rustTodo
syn region    rustComment     start="//" end="$" contains=rustTodo keepend
syn region    rustCommentMLDoc start="/\*\%(!\|\*/\@!\)" end="\*/" contains=rustTodo
syn region    rustCommentDoc  start="//[/!]" end="$" contains=rustTodo keepend

syn keyword rustTodo contained TODO FIXME XXX NB NOTE

" Folding rules {{{2
" Trivial folding rules to begin with.
" TODO: use the AST to make really good folding
syn region rustFoldBraces start="{" end="}" transparent fold
" If you wish to enable this, setlocal foldmethod=syntax
" It's not enabled by default as it would drive some people mad.

" Default highlighting {{{1
hi def link rustHexNumber       rustNumber
hi def link rustBinNumber       rustNumber
hi def link rustIdentifierPrime rustIdentifier
hi def link rustTrait           rustType

hi def link rustSigil         StorageClass
hi def link rustFormat        Special
hi def link rustSpecial       Special
hi def link rustStringContinuation Special
hi def link rustString        String
hi def link rustCharacter     Character
hi def link rustNumber        Number
hi def link rustBoolean       Boolean
hi def link rustEnum          rustType
hi def link rustEnumVariant   rustConstant
hi def link rustConstant      Constant
hi def link rustSelf          Constant
hi def link rustFloat         Float
hi def link rustOperator      Operator
hi def link rustKeyword       Keyword
hi def link rustConditional   Conditional
hi def link rustIdentifier    Identifier
hi def link rustCapsIdent     rustIdentifier
hi def link rustModPath       Include
hi def link rustModPathSep    Delimiter
hi def link rustFunction      Function
hi def link rustFuncName      Function
hi def link rustFuncCall      Function
hi def link rustCommentMLDoc  rustCommentDoc
hi def link rustCommentDoc    SpecialComment
hi def link rustCommentML     rustComment
hi def link rustComment       Comment
hi def link rustAssert        PreCondit
hi def link rustFail          PreCondit
hi def link rustMacro         Macro
hi def link rustType          Type
hi def link rustTodo          Todo
hi def link rustAttribute     PreProc
hi def link rustDeriving      PreProc
hi def link rustStorage       StorageClass
hi def link rustLifetime      Special

" Other Suggestions:
" hi rustAttribute ctermfg=cyan
" hi rustDeriving ctermfg=cyan
" hi rustAssert ctermfg=yellow
" hi rustFail ctermfg=red
" hi rustMacro ctermfg=magenta

syn sync minlines=200
syn sync maxlines=500

let b:current_syntax = "rust"
