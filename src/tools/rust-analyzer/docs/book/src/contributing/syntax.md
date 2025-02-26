# Syntax in rust-analyzer

## About the guide

This guide describes the current state of syntax trees and parsing in rust-analyzer as of 2020-01-09 ([link to commit](https://github.com/rust-lang/rust-analyzer/tree/cf5bdf464cad7ceb9a67e07985a3f4d3799ec0b6)).

## Source Code

The things described are implemented in three places

* [rowan](https://github.com/rust-analyzer/rowan/tree/v0.15.10) -- a generic library for rowan syntax trees.
* [syntax](https://github.com/rust-lang/rust-analyzer/tree/36a70b7435c48837018c71576d7bb4e8f763f501/crates/syntax) crate inside rust-analyzer which wraps `rowan` into rust-analyzer specific API.
  Nothing in rust-analyzer except this crate knows about `rowan`.
* [parser](https://github.com/rust-lang/rust-analyzer/tree/36a70b7435c48837018c71576d7bb4e8f763f501/crates/parser) crate parses input tokens into a `syntax` tree

## Design Goals

* Syntax trees are lossless, or full fidelity. All comments and whitespace get preserved.
* Syntax trees are semantic-less. They describe *strictly* the structure of a sequence of characters, they don't have hygiene, name resolution or type information attached.
* Syntax trees are simple value types. It is possible to create trees for a syntax without any external context.
* Syntax trees have intuitive traversal API (parent, children, siblings, etc).
* Parsing is lossless (even if the input is invalid, the tree produced by the parser represents it exactly).
* Parsing is resilient (even if the input is invalid, parser tries to see as much syntax tree fragments in the input as it can).
* Performance is important, it's OK to use `unsafe` if it means better memory/cpu usage.
* Keep the parser and the syntax tree isolated from each other, such that they can vary independently.

## Trees

### Overview

The syntax tree consists of three layers:

* GreenNodes
* SyntaxNodes (aka RedNode)
* AST

Of these, only GreenNodes store the actual data, the other two layers are (non-trivial) views into green tree.
Red-green terminology comes from Roslyn ([link](https://ericlippert.com/2012/06/08/red-green-trees/)) and gives the name to the `rowan` library. Green and syntax nodes are defined in rowan, ast is defined in rust-analyzer.

Syntax trees are a semi-transient data structure.
In general, frontend does not keep syntax trees for all files in memory.
Instead, it *lowers* syntax trees to more compact and rigid representation, which is not full-fidelity, but which can be mapped back to a syntax tree if so desired.

### GreenNode

GreenNode is a purely-functional tree with arbitrary arity. Conceptually, it is equivalent to the following run of the mill struct:

```rust
#[derive(PartialEq, Eq, Clone, Copy)]
struct SyntaxKind(u16);

#[derive(PartialEq, Eq, Clone)]
struct Node {
    kind: SyntaxKind,
    text_len: usize,
    children: Vec<Arc<Either<Node, Token>>>,
}

#[derive(PartialEq, Eq, Clone)]
struct Token {
    kind: SyntaxKind,
    text: String,
}
```

All the difference between the above sketch and the real implementation are strictly due to optimizations.

Points of note:
* The tree is untyped. Each node has a "type tag", `SyntaxKind`.
* Interior and leaf nodes are distinguished on the type level.
* Trivia and non-trivia tokens are not distinguished on the type level.
* Each token carries its full text.
* The original text can be recovered by concatenating the texts of all tokens in order.
* Accessing a child of particular type (for example, parameter list of a function) generally involves linearly traversing the children, looking for a specific `kind`.
* Modifying the tree is roughly `O(depth)`.
  We don't make special efforts to guarantee that the depth is not linear, but, in practice, syntax trees are branchy and shallow.
* If mandatory (grammar wise) node is missing from the input, it's just missing from the tree.
* If an extra erroneous input is present, it is wrapped into a node with `ERROR` kind, and treated just like any other node.
* Parser errors are not a part of syntax tree.

An input like `fn f() { 90 + 2 }` might be parsed as

```
FN@0..17
  FN_KW@0..2 "fn"
  WHITESPACE@2..3 " "
  NAME@3..4
    IDENT@3..4 "f"
  PARAM_LIST@4..6
    L_PAREN@4..5 "("
    R_PAREN@5..6 ")"
  WHITESPACE@6..7 " "
  BLOCK_EXPR@7..17
    L_CURLY@7..8 "{"
    WHITESPACE@8..9 " "
    BIN_EXPR@9..15
      LITERAL@9..11
        INT_NUMBER@9..11 "90"
      WHITESPACE@11..12 " "
      PLUS@12..13 "+"
      WHITESPACE@13..14 " "
      LITERAL@14..15
        INT_NUMBER@14..15 "2"
    WHITESPACE@15..16 " "
    R_CURLY@16..17 "}"
```

#### Optimizations

(significant amount of implementation work here was done by [CAD97](https://github.com/cad97)).

To reduce the amount of allocations, the GreenNode is a [DST](https://doc.rust-lang.org/reference/dynamically-sized-types.html), which uses a single allocation for header and children. Thus, it is only usable behind a pointer.

```
*-----------+------+----------+------------+--------+--------+-----+--------*
| ref_count | kind | text_len | n_children | child1 | child2 | ... | childn |
*-----------+------+----------+------------+--------+--------+-----+--------*
```

To more compactly store the children, we box *both* interior nodes and tokens, and represent
`Either<Arc<Node>, Arc<Token>>` as a single pointer with a tag in the last bit.

To avoid allocating EVERY SINGLE TOKEN on the heap, syntax trees use interning.
Because the tree is fully immutable, it's valid to structurally share subtrees.
For example, in `1 + 1`, there will be a *single* token for `1` with ref count 2; the same goes for the ` ` whitespace token.
Interior nodes are shared as well (for example in `(1 + 1) * (1 + 1)`).

Note that, the result of the interning is an `Arc<Node>`.
That is, it's not an index into interning table, so you don't have to have the table around to do anything with the tree.
Each tree is fully self-contained (although different trees might share parts).
Currently, the interner is created per-file, but it will be easy to use a per-thread or per-some-context one.

We use a `TextSize`, a newtyped `u32`, to store the length of the text.

We currently use `SmolStr`, a small object optimized string to store text.
This was mostly relevant *before* we implemented tree interning, to avoid allocating common keywords and identifiers. We should switch to storing text data alongside the interned tokens.

#### Alternative designs

##### Dealing with trivia

In the above model, whitespace is not treated specially.
Another alternative (used by swift and roslyn) is to explicitly divide the set of tokens into trivia and non-trivia tokens, and represent non-trivia tokens as

```rust
struct Token {
    kind: NonTriviaTokenKind,
    text: String,
    leading_trivia: Vec<TriviaToken>,
    trailing_trivia: Vec<TriviaToken>,
}
```

The tree then contains only non-trivia tokens.

Another approach (from Dart) is to, in addition to a syntax tree, link all the tokens into a bidirectional link list.
That way, the tree again contains only non-trivia tokens.

Explicit trivia nodes, like in `rowan`, are used by IntelliJ.

##### Accessing Children

As noted before, accessing a specific child in the node requires a linear traversal of the children (though we can skip tokens, because the tag is encoded in the pointer itself).
It is possible to recover O(1) access with another representation.
We explicitly store optional and missing (required by the grammar, but not present) nodes.
That is, we use `Option<Node>` for children.
We also remove trivia tokens from the tree.
This way, each child kind generally occupies a fixed position in a parent, and we can use index access to fetch it.
The cost is that we now need to allocate space for all not-present optional nodes.
So, `fn foo() {}` will have slots for visibility, unsafeness, attributes, abi and return type.

IntelliJ uses linear traversal.
Roslyn and Swift do `O(1)` access.

##### Mutable Trees

IntelliJ uses mutable trees.
Overall, it creates a lot of additional complexity.
However, the API for *editing* syntax trees is nice.

For example the assist to move generic bounds to where clause has this code:

```kotlin
 for typeBound in typeBounds {
     typeBound.typeParamBounds?.delete()
}
```

Modeling this with immutable trees is possible, but annoying.

### Syntax Nodes

A function green tree is not super-convenient to use.
The biggest problem is accessing parents (there are no parent pointers!).
But there are also "identify" issues.
Let's say you want to write a code which builds a list of expressions in a file: `fn collect_expressions(file: GreenNode) -> HashSet<GreenNode>`.
For the input like

```rust
fn main() {
    let x = 90i8;
    let x = x + 2;
    let x = 90i64;
    let x = x + 2;
}
```

both copies of the `x + 2` expression are representing by equal (and, with interning in mind, actually the same) green nodes.
Green trees just can't differentiate between the two.

`SyntaxNode` adds parent pointers and identify semantics to green nodes.
They can be called cursors or [zippers](https://en.wikipedia.org/wiki/Zipper_(data_structure)) (fun fact: zipper is a derivative (as in ′) of a data structure).

Conceptually, a `SyntaxNode` looks like this:

```rust
type SyntaxNode = Arc<SyntaxData>;

struct SyntaxData {
    offset: usize,
    parent: Option<SyntaxNode>,
    green: Arc<GreenNode>,
}

impl SyntaxNode {
    fn new_root(root: Arc<GreenNode>) -> SyntaxNode {
        Arc::new(SyntaxData {
            offset: 0,
            parent: None,
            green: root,
        })
    }
    fn parent(&self) -> Option<SyntaxNode> {
        self.parent.clone()
    }
    fn children(&self) -> impl Iterator<Item = SyntaxNode> {
        let mut offset = self.offset;
        self.green.children().map(|green_child| {
            let child_offset = offset;
            offset += green_child.text_len;
            Arc::new(SyntaxData {
                offset: child_offset,
                parent: Some(Arc::clone(self)),
                green: Arc::clone(green_child),
            })
        })
    }
}

impl PartialEq for SyntaxNode {
    fn eq(&self, other: &SyntaxNode) -> bool {
        self.offset == other.offset
            && Arc::ptr_eq(&self.green, &other.green)
    }
}
```

Points of note:

* SyntaxNode remembers its parent node (and, transitively, the path to the root of the tree)
* SyntaxNode knows its *absolute* text offset in the whole file
* Equality is based on identity. Comparing nodes from different trees does not make sense.

#### Optimization

The reality is different though :-)
Traversal of trees is a common operation, and it makes sense to optimize it.
In particular, the above code allocates and does atomic operations during a traversal.

To get rid of atomics, `rowan` uses non thread-safe `Rc`.
This is OK because trees traversals mostly (always, in case of rust-analyzer) run on a single thread. If you need to send a `SyntaxNode` to another thread, you can send a pair of **root**`GreenNode` (which is thread safe) and a `Range<usize>`.
The other thread can restore the `SyntaxNode` by traversing from the root green node and looking for a node with specified range.
You can also use the similar trick to store a `SyntaxNode`.
That is, a data structure that holds a `(GreenNode, Range<usize>)` will be `Sync`.
However, rust-analyzer goes even further.
It treats trees as semi-transient and instead of storing a `GreenNode`, it generally stores just the id of the file from which the tree originated: `(FileId, Range<usize>)`.
The `SyntaxNode` is the restored by reparsing the file and traversing it from root.
With this trick, rust-analyzer holds only a small amount of trees in memory at the same time, which reduces memory usage.

Additionally, only the root `SyntaxNode` owns an `Arc` to the (root) `GreenNode`.
All other `SyntaxNode`s point to corresponding `GreenNode`s with a raw pointer.
They also point to the parent (and, consequently, to the root) with an owning `Rc`, so this is sound.
In other words, one needs *one* arc bump when initiating a traversal.

To get rid of allocations, `rowan` takes advantage of `SyntaxNode: !Sync` and uses a thread-local free list of `SyntaxNode`s.
In a typical traversal, you only directly hold a few `SyntaxNode`s at a time (and their ancestors indirectly), so a free list proportional to the depth of the tree removes all allocations in a typical case.

So, while traversal is not exactly incrementing a pointer, it's still pretty cheap: TLS + rc bump!

Traversal also yields (cheap) owned nodes, which improves ergonomics quite a bit.

#### Alternative Designs

##### Memoized RedNodes

C# and Swift follow the design where the red nodes are memoized, which would look roughly like this in Rust:

```rust
type SyntaxNode = Arc<SyntaxData>;

struct SyntaxData {
    offset: usize,
    parent: Option<SyntaxNode>,
    green: Arc<GreenNode>,
    children: Vec<OnceCell<SyntaxNode>>,
}
```

This allows using true pointer equality for comparison of identities of `SyntaxNodes`.
rust-analyzer used to have this design as well, but we've since switched to cursors.
The main problem with memoizing the red nodes is that it more than doubles the memory requirements for fully realized syntax trees.
In contrast, cursors generally retain only a path to the root.
C# combats increased memory usage by using weak references.

### AST

`GreenTree`s are untyped and homogeneous, because it makes accommodating error nodes, arbitrary whitespace and comments natural, and because it makes possible to write generic tree traversals.
However, when working with a specific node, like a function definition, one would want a strongly typed API.

This is what is provided by the AST layer. AST nodes are transparent wrappers over untyped syntax nodes:

```rust
pub trait AstNode {
    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized;

    fn syntax(&self) -> &SyntaxNode;
}
```

Concrete nodes are generated (there are 117 of them), and look roughly like this:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FnDef {
    syntax: SyntaxNode,
}

impl AstNode for FnDef {
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        match kind {
            FN => Some(FnDef { syntax }),
            _ => None,
        }
    }
    fn syntax(&self) -> &SyntaxNode {
        &self.syntax
    }
}

impl FnDef {
    pub fn param_list(&self) -> Option<ParamList> {
        self.syntax.children().find_map(ParamList::cast)
    }
    pub fn ret_type(&self) -> Option<RetType> {
        self.syntax.children().find_map(RetType::cast)
    }
    pub fn body(&self) -> Option<BlockExpr> {
        self.syntax.children().find_map(BlockExpr::cast)
    }
    // ...
}
```

Variants like expressions, patterns or items are modeled with `enum`s, which also implement `AstNode`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssocItem {
    FnDef(FnDef),
    TypeAliasDef(TypeAliasDef),
    ConstDef(ConstDef),
}

impl AstNode for AssocItem {
    ...
}
```

Shared AST substructures are modeled via (dynamically compatible) traits:

```rust
trait HasVisibility: AstNode {
    fn visibility(&self) -> Option<Visibility>;
}

impl HasVisibility for FnDef {
    fn visibility(&self) -> Option<Visibility> {
        self.syntax.children().find_map(Visibility::cast)
    }
}
```

Points of note:

* Like `SyntaxNode`s, AST nodes are cheap to clone pointer-sized owned values.
* All "fields" are optional, to accommodate incomplete and/or erroneous source code.
* It's always possible to go from an ast node to an untyped `SyntaxNode`.
* It's possible to go in the opposite direction with a checked cast.
* `enum`s allow modeling of arbitrary intersecting subsets of AST types.
* Most of rust-analyzer works with the ast layer, with notable exceptions of:
  * macro expansion, which needs access to raw tokens and works with `SyntaxNode`s
  * some IDE-specific features like syntax highlighting are more conveniently implemented over a homogeneous `SyntaxNode` tree

#### Alternative Designs

##### Semantic Full AST

In IntelliJ the AST layer (dubbed **P**rogram **S**tructure **I**nterface) can have semantics attached, and is usually backed by either syntax tree, indices, or metadata from compiled libraries.
The backend for PSI can change dynamically.

### Syntax Tree Recap

At its core, the syntax tree is a purely functional n-ary tree, which stores text at the leaf nodes and node "kinds" at all nodes.
A cursor layer is added on top, which gives owned, cheap to clone nodes with identity semantics, parent links and absolute offsets.
An AST layer is added on top, which reifies each node `Kind` as a separate Rust type with the corresponding API.

## Parsing

The (green) tree is constructed by a DFS "traversal" of the desired tree structure:

```rust
pub struct GreenNodeBuilder { ... }

impl GreenNodeBuilder {
    pub fn new() -> GreenNodeBuilder { ... }

    pub fn token(&mut self, kind: SyntaxKind, text: &str) { ... }

    pub fn start_node(&mut self, kind: SyntaxKind) { ... }
    pub fn finish_node(&mut self) { ... }

    pub fn finish(self) -> GreenNode { ... }
}
```

The parser, ultimately, needs to invoke the `GreenNodeBuilder`.
There are two principal sources of inputs for the parser:
  * source text, which contains trivia tokens (whitespace and comments)
  * token trees from macros, which lack trivia

Additionally, input tokens do not correspond 1-to-1 with output tokens.
For example, two consecutive `>` tokens might be glued, by the parser, into a single `>>`.

For these reasons, the parser crate defines a callback interfaces for both input tokens and output trees.
The explicit glue layer then bridges various gaps.

The parser interface looks like this:

```rust
pub struct Token {
    pub kind: SyntaxKind,
    pub is_joined_to_next: bool,
}

pub trait TokenSource {
    fn current(&self) -> Token;
    fn lookahead_nth(&self, n: usize) -> Token;
    fn is_keyword(&self, kw: &str) -> bool;

    fn bump(&mut self);
}

pub trait TreeSink {
    fn token(&mut self, kind: SyntaxKind, n_tokens: u8);

    fn start_node(&mut self, kind: SyntaxKind);
    fn finish_node(&mut self);

    fn error(&mut self, error: ParseError);
}

pub fn parse(
    token_source: &mut dyn TokenSource,
    tree_sink: &mut dyn TreeSink,
) { ... }
```

Points of note:

* The parser and the syntax tree are independent, they live in different crates neither of which depends on the other.
* The parser doesn't know anything about textual contents of the tokens, with an isolated hack for checking contextual keywords.
* For gluing tokens, the `TreeSink::token` might advance further than one atomic token ahead.

### Reporting Syntax Errors

Syntax errors are not stored directly in the tree.
The primary motivation for this is that syntax tree is not necessary produced by the parser, it may also be assembled manually from pieces (which happens all the time in refactorings).
Instead, parser reports errors to an error sink, which stores them in a `Vec`.
If possible, errors are not reported during parsing and are postponed for a separate validation step.
For example, parser accepts visibility modifiers on trait methods, but then a separate tree traversal flags all such visibilities as erroneous.

### Macros

The primary difficulty with macros is that individual tokens have identities, which need to be preserved in the syntax tree for hygiene purposes.
This is handled by the `TreeSink` layer.
Specifically, `TreeSink` constructs the tree in lockstep with draining the original token stream.
In the process, it records which tokens of the tree correspond to which tokens of the input, by using text ranges to identify syntax tokens.
The end result is that parsing an expanded code yields a syntax tree and a mapping of text-ranges of the tree to original tokens.

To deal with precedence in cases like `$expr * 1`, we use special invisible parenthesis, which are explicitly handled by the parser.

### Whitespace & Comments

Parser does not see whitespace nodes.
Instead, they are attached to the tree in the `TreeSink` layer.

For example, in

```rust
// non doc comment
fn foo() {}
```

the comment will be (heuristically) made a child of function node.

### Incremental Reparse

Green trees are cheap to modify, so incremental reparse works by patching a previous tree, without maintaining any additional state.
The reparse is based on heuristic: we try to contain a change to a single `{}` block, and reparse only this block.
To do this, we maintain the invariant that, even for invalid code, curly braces are always paired correctly.

In practice, incremental reparsing doesn't actually matter much for IDE use-cases, parsing from scratch seems to be fast enough.

### Parsing Algorithm

We use a boring hand-crafted recursive descent + pratt combination, with a special effort of continuing the parsing if an error is detected.

### Parser Recap

Parser itself defines traits for token sequence input and syntax tree output.
It doesn't care about where the tokens come from, and how the resulting syntax tree looks like.
