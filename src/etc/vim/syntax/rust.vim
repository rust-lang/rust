" Vim syntax file
" Language:     Rust
" Maintainer:   Patrick Walton <pcwalton@mozilla.com>
" Maintainer:   Ben Blum <bblum@cs.cmu.edu>
" Maintainer:   Chris Morgan <me@chrismorgan.info>
" Last Change:  2014 Feb 27

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
syn keyword   rustKeyword     break
syn keyword   rustKeyword     box nextgroup=rustBoxPlacement skipwhite skipempty
syn keyword   rustKeyword     continue
syn keyword   rustKeyword     extern nextgroup=rustExternCrate,rustObsoleteExternMod skipwhite skipempty
syn keyword   rustKeyword     fn nextgroup=rustFuncName skipwhite skipempty
syn keyword   rustKeyword     for in if impl let
syn keyword   rustKeyword     loop once proc pub
syn keyword   rustKeyword     return super
syn keyword   rustKeyword     unsafe virtual while
syn keyword   rustKeyword     use nextgroup=rustModPath skipwhite skipempty
" FIXME: Scoped impl's name is also fallen in this category
syn keyword   rustKeyword     mod trait struct enum type nextgroup=rustIdentifier skipwhite skipempty
syn keyword   rustStorage     mut ref static const

syn keyword   rustInvalidBareKeyword crate

syn keyword   rustExternCrate crate contained nextgroup=rustIdentifier skipwhite skipempty
syn keyword   rustObsoleteExternMod mod contained nextgroup=rustIdentifier skipwhite skipempty

syn match     rustIdentifier  contains=rustIdentifierPrime "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained
syn match     rustFuncName    "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained

syn region    rustBoxPlacement matchgroup=rustBoxPlacementParens start="(" end=")" contains=TOP contained
syn keyword   rustBoxPlacementExpr GC containedin=rustBoxPlacement
" Ideally we'd have syntax rules set up to match arbitrary expressions. Since
" we don't, we'll just define temporary contained rules to handle balancing
" delimiters.
syn region    rustBoxPlacementBalance start="(" end=")" containedin=rustBoxPlacement transparent
syn region    rustBoxPlacementBalance start="\[" end="\]" containedin=rustBoxPlacement transparent
" {} are handled by rustFoldBraces

" Reserved (but not yet used) keywords {{{2
syn keyword   rustReservedKeyword alignof be do offsetof priv pure sizeof typeof unsized yield

" Built-in types {{{2
syn keyword   rustType        int uint float char bool u8 u16 u32 u64 f32
syn keyword   rustType        f64 i8 i16 i32 i64 str Self

" Things from the prelude (src/libstd/prelude.rs) {{{2
" This section is just straight transformation of the contents of the prelude,
" to make it easy to update.

" Core operators {{{3
syn keyword   rustTrait       Copy Send Sized Share
syn keyword   rustTrait       Add Sub Mul Div Rem Neg Not
syn keyword   rustTrait       BitAnd BitOr BitXor
syn keyword   rustTrait       Drop Deref DerefMut
syn keyword   rustTrait       Shl Shr Index IndexMut
syn keyword   rustEnum        Option
syn keyword   rustEnumVariant Some None
syn keyword   rustEnum        Result
syn keyword   rustEnumVariant Ok Err

" Functions {{{3
"syn keyword rustFunction from_str
"syn keyword rustFunction range
"syn keyword rustFunction drop

" Types and traits {{{3
syn keyword rustTrait Ascii AsciiCast OwnedAsciiCast AsciiStr
syn keyword rustTrait IntoBytes
syn keyword rustTrait ToCStr
syn keyword rustTrait Char
syn keyword rustTrait Clone
syn keyword rustTrait PartialEq PartialOrd Eq Ord Equiv
syn keyword rustEnum Ordering
syn keyword rustEnumVariant Less Equal Greater
syn keyword rustTrait Collection Mutable Map MutableMap
syn keyword rustTrait Set MutableSet
syn keyword rustTrait FromIterator Extendable ExactSize
syn keyword rustTrait Iterator DoubleEndedIterator
syn keyword rustTrait RandomAccessIterator CloneableIterator
syn keyword rustTrait OrdIterator MutableDoubleEndedIterator
syn keyword rustTrait Num NumCast CheckedAdd CheckedSub CheckedMul CheckedDiv
syn keyword rustTrait Signed Unsigned Primitive Int Float
syn keyword rustTrait FloatMath ToPrimitive FromPrimitive
syn keyword rustTrait Box
syn keyword rustTrait GenericPath Path PosixPath WindowsPath
syn keyword rustTrait RawPtr
syn keyword rustTrait Buffer Writer Reader Seek
syn keyword rustTrait Str StrVector StrSlice OwnedStr
syn keyword rustTrait IntoMaybeOwned StrAllocating
syn keyword rustTrait ToStr IntoStr
syn keyword rustTrait Tuple1 Tuple2 Tuple3 Tuple4
syn keyword rustTrait Tuple5 Tuple6 Tuple7 Tuple8
syn keyword rustTrait Tuple9 Tuple10 Tuple11 Tuple12
syn keyword rustTrait CloneableVector ImmutableCloneableVector
syn keyword rustTrait MutableCloneableVector MutableOrdVector
syn keyword rustTrait ImmutableVector MutableVector
syn keyword rustTrait ImmutableEqVector ImmutableOrdVector
syn keyword rustTrait Vector VectorVector
syn keyword rustTrait MutableVectorAllocating
syn keyword rustTrait String
syn keyword rustTrait Vec

"syn keyword rustFunction sync_channel channel
syn keyword rustTrait SyncSender Sender Receiver
"syn keyword rustFunction spawn

"syn keyword rustConstant GC

syn keyword   rustSelf        self
syn keyword   rustBoolean     true false

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

syn match     rustEscapeError   display contained /\\./
syn match     rustEscape        display contained /\\\([nrt0\\'"]\|x\x\{2}\)/
syn match     rustEscapeUnicode display contained /\\\(u\x\{4}\|U\x\{8}\)/
syn match     rustStringContinuation display contained /\\\n\s*/
syn region    rustString      start=+b"+ skip=+\\\\\|\\"+ end=+"+ contains=rustEscape,rustEscapeError,rustStringContinuation
syn region    rustString      start=+"+ skip=+\\\\\|\\"+ end=+"+ contains=rustEscape,rustEscapeUnicode,rustEscapeError,rustStringContinuation,@Spell
syn region    rustString      start='b\?r\z(#*\)"' end='"\z1' contains=@Spell

syn region    rustAttribute   start="#!\?\[" end="\]" contains=rustString,rustDeriving
syn region    rustDeriving    start="deriving(" end=")" contained contains=rustTrait

" Number literals
syn match     rustDecNumber   display "\<[0-9][0-9_]*\%([iu]\%(8\|16\|32\|64\)\=\)\="
syn match     rustHexNumber   display "\<0x[a-fA-F0-9_]\+\%([iu]\%(8\|16\|32\|64\)\=\)\="
syn match     rustOctNumber   display "\<0o[0-7_]\+\%([iu]\%(8\|16\|32\|64\)\=\)\="
syn match     rustBinNumber   display "\<0b[01_]\+\%([iu]\%(8\|16\|32\|64\)\=\)\="

" Special case for numbers of the form "1." which are float literals, unless followed by
" an identifier, which makes them integer literals with a method call or field access.
" (This must go first so the others take precedence.)
syn match     rustFloat       display "\<[0-9][0-9_]*\.\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\@!"
" To mark a number as a normal float, it must have at least one of the three things integral values don't have:
" a decimal point and more numbers; an exponent; and a type suffix.
syn match     rustFloat       display "\<[0-9][0-9_]*\%(\.[0-9][0-9_]*\)\%([eE][+-]\=[0-9_]\+\)\=\(f32\|f64\)\="
syn match     rustFloat       display "\<[0-9][0-9_]*\%(\.[0-9][0-9_]*\)\=\%([eE][+-]\=[0-9_]\+\)\(f32\|f64\)\="
syn match     rustFloat       display "\<[0-9][0-9_]*\%(\.[0-9][0-9_]*\)\=\%([eE][+-]\=[0-9_]\+\)\=\(f32\|f64\)"

" For the benefit of delimitMate
syn region rustLifetimeCandidate display start=/&'\%(\([^'\\]\|\\\(['nrt0\\\"]\|x\x\{2}\|u\x\{4}\|U\x\{8}\)\)'\)\@!/ end=/[[:cntrl:][:space:][:punct:]]\@=\|$/ contains=rustSigil,rustLifetime
syn region rustGenericRegion display start=/<\%('\|[^[cntrl:][:space:][:punct:]]\)\@=')\S\@=/ end=/>/ contains=rustGenericLifetimeCandidate
syn region rustGenericLifetimeCandidate display start=/\%(<\|,\s*\)\@<='/ end=/[[:cntrl:][:space:][:punct:]]\@=\|$/ contains=rustSigil,rustLifetime

"rustLifetime must appear before rustCharacter, or chars will get the lifetime highlighting
syn match     rustLifetime    display "\'\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*"
syn match   rustCharacterInvalid   display contained /b\?'\zs[\n\r\t']\ze'/
" The groups negated here add up to 0-255 but nothing else (they do not seem to go beyond ASCII).
syn match   rustCharacterInvalidUnicode   display contained /b'\zs[^[:cntrl:][:graph:][:alnum:][:space:]]\ze'/
syn match   rustCharacter   /b'\([^\\]\|\\\(.\|x\x\{2}\)\)'/ contains=rustEscape,rustEscapeError,rustCharacterInvalid,rustCharacterInvalidUnicode
syn match   rustCharacter   /'\([^\\]\|\\\(.\|x\x\{2}\|u\x\{4}\|U\x\{8}\)\)'/ contains=rustEscape,rustEscapeUnicode,rustEscapeError,rustCharacterInvalid

syn region rustCommentLine                                        start="//"                      end="$"   contains=rustTodo,@Spell
syn region rustCommentLineDoc                                     start="//\%(//\@!\|!\)"         end="$"   contains=rustTodo,@Spell
syn region rustCommentBlock    matchgroup=rustCommentBlock        start="/\*\%(!\|\*[*/]\@!\)\@!" end="\*/" contains=rustTodo,rustCommentBlockNest,@Spell
syn region rustCommentBlockDoc matchgroup=rustCommentBlockDoc     start="/\*\%(!\|\*[*/]\@!\)"    end="\*/" contains=rustTodo,rustCommentBlockDocNest,@Spell
syn region rustCommentBlockNest matchgroup=rustCommentBlock       start="/\*"                     end="\*/" contains=rustTodo,rustCommentBlockNest,@Spell contained transparent
syn region rustCommentBlockDocNest matchgroup=rustCommentBlockDoc start="/\*"                     end="\*/" contains=rustTodo,rustCommentBlockDocNest,@Spell contained transparent
" FIXME: this is a really ugly and not fully correct implementation. Most
" importantly, a case like ``/* */*`` should have the final ``*`` not being in
" a comment, but in practice at present it leaves comments open two levels
" deep. But as long as you stay away from that particular case, I *believe*
" the highlighting is correct. Due to the way Vim's syntax engine works
" (greedy for start matches, unlike Rust's tokeniser which is searching for
" the earliest-starting match, start or end), I believe this cannot be solved.
" Oh you who would fix it, don't bother with things like duplicating the Block
" rules and putting ``\*\@<!`` at the start of them; it makes it worse, as
" then you must deal with cases like ``/*/**/*/``. And don't try making it
" worse with ``\%(/\@<!\*\)\@<!``, either...

syn keyword rustTodo contained TODO FIXME XXX NB NOTE

" Folding rules {{{2
" Trivial folding rules to begin with.
" TODO: use the AST to make really good folding
syn region rustFoldBraces start="{" end="}" transparent fold
" If you wish to enable this, setlocal foldmethod=syntax
" It's not enabled by default as it would drive some people mad.

" Default highlighting {{{1
hi def link rustDecNumber       rustNumber
hi def link rustHexNumber       rustNumber
hi def link rustOctNumber       rustNumber
hi def link rustBinNumber       rustNumber
hi def link rustIdentifierPrime rustIdentifier
hi def link rustTrait           rustType

hi def link rustSigil         StorageClass
hi def link rustEscape        Special
hi def link rustEscapeUnicode rustEscape
hi def link rustEscapeError   Error
hi def link rustStringContinuation Special
hi def link rustString        String
hi def link rustCharacterInvalid Error
hi def link rustCharacterInvalidUnicode rustCharacterInvalid
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
hi def link rustReservedKeyword Error
hi def link rustConditional   Conditional
hi def link rustIdentifier    Identifier
hi def link rustCapsIdent     rustIdentifier
hi def link rustModPath       Include
hi def link rustModPathSep    Delimiter
hi def link rustFunction      Function
hi def link rustFuncName      Function
hi def link rustFuncCall      Function
hi def link rustCommentLine   Comment
hi def link rustCommentLineDoc SpecialComment
hi def link rustCommentBlock  rustCommentLine
hi def link rustCommentBlockDoc rustCommentLineDoc
hi def link rustAssert        PreCondit
hi def link rustFail          PreCondit
hi def link rustMacro         Macro
hi def link rustType          Type
hi def link rustTodo          Todo
hi def link rustAttribute     PreProc
hi def link rustDeriving      PreProc
hi def link rustStorage       StorageClass
hi def link rustObsoleteStorage Error
hi def link rustLifetime      Special
hi def link rustInvalidBareKeyword Error
hi def link rustExternCrate   rustKeyword
hi def link rustObsoleteExternMod Error
hi def link rustBoxPlacementParens Delimiter
hi def link rustBoxPlacementExpr rustKeyword

" Other Suggestions:
" hi rustAttribute ctermfg=cyan
" hi rustDeriving ctermfg=cyan
" hi rustAssert ctermfg=yellow
" hi rustFail ctermfg=red
" hi rustMacro ctermfg=magenta

syn sync minlines=200
syn sync maxlines=500

let b:current_syntax = "rust"
