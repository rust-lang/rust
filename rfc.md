- Feature Name: libsyntax2.0
- Start Date: 2017-12-30
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)


>I think the lack of reusability comes in object-oriented languages,
>not functional languages. Because the problem with object-oriented
>languages is they’ve got all this implicit environment that they
>carry around with them. You wanted a banana but what you got was a
>gorilla holding the banana and the entire jungle.
>
>If you have referentially transparent code, if you have pure
>functions — all the data comes in its input arguments and everything
>goes out and leave no state behind — it’s incredibly reusable.
>
> **Joe Armstrong**

# Summary
[summary]: #summary

The long-term plan is to rewrite libsyntax parser and syntax tree data
structure to create a software component independent of the rest of
rustc compiler and suitable for the needs of IDEs and code
editors. This RFCs is the first step of this plan, whose goal is to
find out if this is possible at least in theory. If it is possible,
the next steps would be a prototype implementation as a crates.io
crate and a separate RFC for integrating the prototype with rustc,
other tools, and eventual libsyntax removal.

Note that this RFC does not propose to stabilize any API for working
with rust syntax: the semver version of the hypothetical library would
be `0.1.0`.


# Motivation
[motivation]: #motivation

## Reusability

In theory, the parser can be a pure function, which takes a `&str` as
an input, and produces a `ParseTree` as an output.

This is great for reusability: for example, you can compile this
function to WASM and use it for fast client-side validation of syntax
on the rust playground, or you can develop tools like `rustfmt` on
stable Rust outside of rustc repository, or you can embed the parser
into your favorite IDE or code editor.

This is also great for correctness: with such simple interface, it's
possible to write property-based tests to thoroughly compare two
different implementations of the parser. It's also straightforward to
create a comprehensive test suite, because all the inputs and outputs
are trivially serializable to human-readable text.

Another benefit is performance: with this signature, you can cache a
parse tree for each file, with trivial strategy for cache invalidation
(invalidate an entry when the underling file changes). On top of such
a cache it is possible to build a smart code indexer which maintains
the set of symbols in the project, watches files for changes and
automatically reindexes only changed files.

Unfortunately, the current libsyntax is far from this ideal. For
example, even the lexer makes use of the `FileMap` which is
essentially a global state of the compiler which represents all know
files. As a data point, it turned out to be easier to move `rustfmt`
into the main `rustc` repository than to move libsyntax outside!


## IDE support

There is one big difference in how IDEs and compilers typically treat
source code.

In the compiler, it is convenient to transform the source
code into Abstract Syntax Tree form, which is independent of the
surface syntax. For example, it's convenient to discard comments,
whitespaces and desugar some syntactic constructs in terms of the
simpler ones.

In contrast, for IDEs it is crucial to have a lossless view of the
source code because, for example, it's important to preserve comments
during refactorings. Ideally, IDEs should be able to incrementally
relex and reparse the file as the user types, because syntax tree is
necessary to correctly handle certain code-editing actions like
autoindentation or joining lines. IDE also must be able to produce
partial parse trees when some input is missing or invalid.

Currently rustc uses the AST approach, and preserves some of the
source code information in the form of spans in the AST.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Not applicable.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This section proposes a new syntax tree data structure, which should
be suitable for both compiler and IDE. It is heavily inspired by [PSI]
data structure which used in [IntelliJ] based IDEs and in the [Kotlin]
compiler.


[PSI]: http://www.jetbrains.org/intellij/sdk/docs/reference_guide/custom_language_support/implementing_parser_and_psi.html
[IntelliJ]: https://github.com/JetBrains/intellij-community/
[Kotlin]: https://kotlinlang.org/


## Untyped Tree

The main idea is to store the minimal amount of information in the
tree itself, and instead lean heavily on the source code for the
actual data about identifier names, constant values etc.

All nodes in the tree are of the same type and store a constant for
the syntactic category of the element and a range in the source code.

Here is a minimal implementation of this data structure with some Rust
syntactic categories


```rust
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeKind(u16);

pub struct File {
	text: String,
	nodes: Vec<NodeData>,
}

struct NodeData {
	kind: NodeKind,
	range: (u32, u32),
	parent: Option<u32>,
	first_child: Option<u32>,
	next_sibling: Option<u32>,
}

#[derive(Clone, Copy)]
pub struct Node<'f> {
	file: &'f File,
	idx: u32,
}

pub struct Children<'f> {
	next: Option<Node<'f>>,
}

impl File {
	pub fn root<'f>(&'f self) -> Node<'f> {
		assert!(!self.nodes.is_empty());
		Node { file: self, idx: 0 }
	}
}

impl<'f> Node<'f> {
	pub fn kind(&self) -> NodeKind {
		self.data().kind
	}

	pub fn text(&self) -> &'f str {
		let (start, end) = self.data().range;
		&self.file.text[start as usize..end as usize]
	}

	pub fn parent(&self) -> Option<Node<'f>> {
		self.as_node(self.data().parent)
	}

	pub fn children(&self) -> Children<'f> {
		Children { next: self.as_node(self.data().first_child) }
	}

	fn data(&self) -> &'f NodeData {
		&self.file.nodes[self.idx as usize]
	}

	fn as_node(&self, idx: Option<u32>) -> Option<Node<'f>> {
		idx.map(|idx| Node { file: self.file, idx })
	}
}

impl<'f> Iterator for Children<'f> {
	type Item = Node<'f>;

	fn next(&mut self) -> Option<Node<'f>> {
		let next = self.next;
		self.next = next.and_then(|node| node.as_node(node.data().next_sibling));
		next
	}
}

pub const ERROR: NodeKind = NodeKind(0);
pub const WHITESPACE: NodeKind = NodeKind(1);
pub const STRUCT_KW: NodeKind = NodeKind(2);
pub const IDENT: NodeKind = NodeKind(3);
pub const L_CURLY: NodeKind = NodeKind(4);
pub const R_CURLY: NodeKind = NodeKind(5);
pub const COLON: NodeKind = NodeKind(6);
pub const COMMA: NodeKind = NodeKind(7);
pub const AMP: NodeKind = NodeKind(8);
pub const LINE_COMMENT: NodeKind = NodeKind(9);
pub const FILE: NodeKind = NodeKind(10);
pub const STRUCT_DEF: NodeKind = NodeKind(11);
pub const FIELD_DEF: NodeKind = NodeKind(12);
pub const TYPE_REF: NodeKind = NodeKind(13);
```

Here is a rust snippet and the corresponding parse tree:

```rust
struct Foo {
	field1: u32,
	&
	// non-doc comment
	field2:
}
```


```
FILE
  STRUCT_DEF
    STRUCT_KW
    WHITESPACE
    IDENT
    WHITESPACE
    L_CURLY
    WHITESPACE
    FIELD_DEF
      IDENT
      COLON
      WHITESPACE
      TYPE_REF
        IDENT
    COMMA
    WHITESPACE
    ERROR
      AMP
    WHITESPACE
    FIELD_DEF
      LINE_COMMENT
      WHITESPACE
      IDENT
      COLON
      ERROR
    WHITESPACE
    R_CURLY
```

Note several features of the tree:

* All whitespace and comments are explicitly accounted for.

* The node for `STRUCT_DEF` contains the error element for `&`, but
  still represents the following field correctly.

* The second field of the struct is incomplete: `FIELD_DEF` node for
  it contains an `ERROR` element, but nevertheless has the correct
  `NodeKind`.

* The non-documenting comment is correctly attached to the following
  field.


## Typed Tree

It's hard to work with this raw parse tree, because it is untyped:
node containing a struct definition has the same API as the node for
the struct field. But it's possible to add a strongly typed layer on
top of this raw tree, and get a zero-cost AST. Here is an example
which adds type-safe wrappers for structs and fields:

```rust
// generic infrastructure

pub trait AstNode<'f>: Copy + 'f {
	fn new(node: Node<'f>) -> Option<Self>;
	fn node(&self) -> Node<'f>;
}

pub fn child_of_kind<'f>(node: Node<'f>, kind: NodeKind) -> Option<Node<'f>> {
	node.children().find(|child| child.kind() == kind)
}

pub fn ast_children<'f, A: AstNode<'f>>(node: Node<'f>) -> Box<Iterator<Item=A> + 'f> {
	Box::new(node.children().filter_map(A::new))
}

// AST elements, specific to Rust

#[derive(Clone, Copy)]
pub struct StructDef<'f>(Node<'f>);

#[derive(Clone, Copy)]
pub struct FieldDef<'f>(Node<'f>);

#[derive(Clone, Copy)]
pub struct TypeRef<'f>(Node<'f>);

pub trait NameOwner<'f>: AstNode<'f> {
	fn name_ident(&self) -> Node<'f> {
		child_of_kind(self.node(), IDENT).unwrap()
	}

	fn name(&self) -> &'f str { self.name_ident().text() }
}


impl<'f> AstNode<'f> for StructDef<'f> {
	fn new(node: Node<'f>) -> Option<Self> {
		if node.kind() == STRUCT_DEF { Some(StructDef(node)) } else { None }
	}
	fn node(&self) -> Node<'f> { self.0 }
}

impl<'f> NameOwner<'f> for StructDef<'f> {}

impl<'f> StructDef<'f> {
	pub fn fields(&self) -> Box<Iterator<Item=FieldDef<'f>> + 'f> {
		ast_children(self.node())
	}
}


impl<'f> AstNode<'f> for FieldDef<'f> {
	fn new(node: Node<'f>) -> Option<Self> {
		if node.kind() == FIELD_DEF { Some(FieldDef(node)) } else { None }
	}
	fn node(&self) -> Node<'f> { self.0 }
}

impl<'f> FieldDef<'f> {
	pub fn type_ref(&self) -> Option<TypeRef<'f>> {
		ast_children(self.node()).next()
	}
}

impl<'f> NameOwner<'f> for FieldDef<'f> {}


impl<'f> AstNode<'f> for TypeRef<'f> {
	fn new(node: Node<'f>) -> Option<Self> {
		if node.kind() == TYPE_REF { Some(TypeRef(node)) } else { None }
	}
	fn node(&self) -> Node<'f> { self.0 }
}
```


## Missing Source Code

The crucial feature of this syntax tree is that it is just a view into
the original source code. And this poses a problem for the Rust
language, because not all compiled Rust code is represented in the
form of source code! Specifically, Rust has a powerful macro system,
which effectively allows to create and parse additional source code at
compile time. It is not entirely clear that the proposed parsing
framework is able to handle this use case, and it's the main purpose
of this RFC to figure it out. The current idea for handling macros is
to make each macro expansion produce a triple of (expansion text,
syntax tree, hygiene information), where hygiene information is a side
table, which colors different ranges of the expansion text according
to the original syntactic context.


## Implementation plan

This RFC proposes huge changes to the internals of the compiler, so
it's important to proceed carefully and incrementally. The following
plan is suggested:

* RFC discussion about the theoretical feasibility of the proposal.

* Implementation of the proposal as a completely separate crates.io
  crate, by refactoring existing libsyntax source code to produce a
  new tree.

* A prototype implementation of the macro expansion on top of the new
  sytnax tree.

* Additional round of discussion/RFC about merging with the mainline
  compiler.


# Drawbacks
[drawbacks]: #drawbacks

- No harm will be done as long as the new libsyntax exists as an
  experiemt on crates.io. However, actually using it in the compiler
  and other tools would require massive refactorings.
  
- Proposed syntax tree requires to keep the original source code
  available, which might increase memory usage of the
  compiler. However, it should be possible to throw the original tree
  and source code away after conversion to HIR.

# Rationale and alternatives
[alternatives]: #alternatives

- Incrementally add more information about source code to the current
  AST.

- Move the current libsyntax to crates.io as is. In the past, there
  were several failed attempts to do that.

- Explore alternative representations for the parse tree.

- Use parser generator instead of hand written parser. Using the
  parser from libsyntax directly would be easier, and hand-written
  LL-style parsers usually have much better error recovery than
  generated LR-style ones.

# Unresolved questions
[unresolved]: #unresolved-questions

- Is it at all possible to represent Rust parser as a pure function of
  the source code? It seems like the answer is yes, because the
  language and especially macros were cleverly designed with this
  use-case in mind.
  
  
- Is it possible to implement macro expansion using the proposed
  framework? This is the main question of this RFC. The proposed
  solution of synthesizing source code on the fly seems workable: it's
  not that different from the current implementation, which
  synthesizes token trees.
  
  
- How to actually phase out current libsyntax, if libsyntax2.0 turns
  out to be a success?
