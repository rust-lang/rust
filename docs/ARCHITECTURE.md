# Design and open questions about libsyntax


The high-level description of the architecture is in RFC.md. You might
also want to dig through https://github.com/matklad/fall/ which
contains some pretty interesting stuff build using similar ideas
(warning: it is completely undocumented, poorly written and in general
not the thing which I recommend to study (yes, this is
self-contradictory)).

## Tree

The centerpiece of this whole endeavor is the syntax tree, in the
`tree` module. Open questions:

- how to best represent errors, to take advantage of the fact that
  they are rare, but to enable fully-persistent style structure
  sharing between tree nodes?
  
- should we make red/green split from Roslyn more pronounced?

- one can layout nodes in a single array in such a way that children
  of the node form a continuous slice. Seems nifty, but do we need it?
  
- should we use SoA or AoS for NodeData?

- should we split leaf nodes and internal nodes into separate arrays?
  Can we use it to save some bits here and there? (leaves don't need
  first_child field, for example).


## Parser

The syntax tree is produced using a three-staged process. 

First, a raw text is split into tokens with a lexer. Lexer has a
peculiar signature: it is an `Fn(&str) -> Token`, where token is a
pair of `SyntaxKind` (you should have read the `tree` module and RFC
by this time! :)) and a len. That is, lexer chomps only the first
token of the input. This forces the lexer to be stateless, and makes
it possible to implement incremental relexing easily.

Then, the bulk of work, the parser turns a stream of tokens into
stream of events. Not that parser **does not** construct a tree right
away. This is done for several reasons:

* to decouple the actual tree data structure from the parser: you can
  build any datastructre you want from the stream of events
  
* to make parsing fast: you can produce a list of events without
  allocations
  
* to make it easy to tweak tree structure. Consider this code:

  ```
  #[cfg(test)]
  pub fn foo() {}
  ```
  
  Here, the attribute and the `pub` keyword must be the children of
  the `fn` node. However, when parsing them, we don't yet know if
  there would be a function ahead: it very well might be a `struct`
  there. If we use events, we generally don't care about this *in
  parser* and just spit them in order.
  
* (Is this true?)  to make incremental reparsing easier: you can reuse
  the same rope data structure for all of the original string, the
  tokens and the events.
  

The parser also does not know about whitespace tokens: it's the job of
the next layer to assign whitespace and comments to nodes. However,
parser can remap contextual tokens, like `>>` or `union`, so it has
access to the text.

And at last, the TreeBuilder converts a flat stream of events into a
tree structure. It also *should* be responsible for attaching comments
and rebalancing the tree, but it does not do this yet :) 


## Error reporing

TODO: describe how stuff like `skip_to_first` works


## Validator

Parser and lexer accept a lot of *invalid* code intentionally. The
idea is to post-process the tree and to proper error reporting,
literal conversion and quick-fix suggestions. There is no
design/implementation for this yet.


## AST

Nothing yet, see `AstNode` in `fall`.
