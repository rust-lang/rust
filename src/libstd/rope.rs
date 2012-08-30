/*!
 * High-level text containers.
 *
 * Ropes are a high-level representation of text that offers
 * much better performance than strings for common operations,
 * and generally reduce memory allocations and copies, while only
 * entailing a small degradation of less common operations.
 *
 * More precisely, where a string is represented as a memory buffer,
 * a rope is a tree structure whose leaves are slices of immutable
 * strings. Therefore, concatenation, appending, prepending, substrings,
 * etc. are operations that require only trivial tree manipulation,
 * generally without having to copy memory. In addition, the tree
 * structure of ropes makes them suitable as a form of index to speed-up
 * access to Unicode characters by index in long chunks of text.
 *
 * The following operations are algorithmically faster in ropes:
 *
 * * extracting a subrope is logarithmic (linear in strings);
 * * appending/prepending is near-constant time (linear in strings);
 * * concatenation is near-constant time (linear in strings);
 * * char length is constant-time (linear in strings);
 * * access to a character by index is logarithmic (linear in strings);
 */

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

/// The type of ropes.
type rope = node::root;

/*
 Section: Creating a rope
 */

/// Create an empty rope
fn empty() -> rope {
   return node::empty;
}

/**
 * Adopt a string as a rope.
 *
 * # Arguments
 *
 * * str - A valid string.
 *
 * # Return value
 *
 * A rope representing the same string as `str`. Depending of the length
 * of `str`, this rope may be empty, flat or complex.
 *
 * # Performance notes
 *
 * * this operation does not copy the string;
 * * the function runs in linear time.
 */
fn of_str(str: @~str) -> rope {
    return of_substr(str, 0u, str::len(*str));
}

/**
 * As `of_str` but for a substring.
 *
 * # Arguments
 * * byte_offset - The offset of `str` at which the rope starts.
 * * byte_len - The number of bytes of `str` to use.
 *
 * # Return value
 *
 * A rope representing the same string as `str::substr(str, byte_offset,
 * byte_len)`.  Depending on `byte_len`, this rope may be empty, flat or
 * complex.
 *
 * # Performance note
 *
 * This operation does not copy the substring.
 *
 * # Safety notes
 *
 * * this function does _not_ check the validity of the substring;
 * * this function fails if `byte_offset` or `byte_len` do not match `str`.
 */
fn of_substr(str: @~str, byte_offset: uint, byte_len: uint) -> rope {
    if byte_len == 0u { return node::empty; }
    if byte_offset + byte_len  > str::len(*str) { fail; }
    return node::content(node::of_substr(str, byte_offset, byte_len));
}

/*
Section: Adding things to a rope
 */

/**
 * Add one char to the end of the rope
 *
 * # Performance note
 *
 * * this function executes in near-constant time
 */
fn append_char(rope: rope, char: char) -> rope {
    return append_str(rope, @str::from_chars(~[char]));
}

/**
 * Add one string to the end of the rope
 *
 * # Performance note
 *
 * * this function executes in near-linear time
 */
fn append_str(rope: rope, str: @~str) -> rope {
    return append_rope(rope, of_str(str))
}

/**
 * Add one char to the beginning of the rope
 *
 * # Performance note
 * * this function executes in near-constant time
 */
fn prepend_char(rope: rope, char: char) -> rope {
    return prepend_str(rope, @str::from_chars(~[char]));
}

/**
 * Add one string to the beginning of the rope
 *
 * # Performance note
 * * this function executes in near-linear time
 */
fn prepend_str(rope: rope, str: @~str) -> rope {
    return append_rope(of_str(str), rope)
}

/// Concatenate two ropes
fn append_rope(left: rope, right: rope) -> rope {
   match (left) {
     node::empty => return right,
     node::content(left_content) => {
       match (right) {
         node::empty => return left,
         node::content(right_content) => {
           return node::content(node::concat2(left_content, right_content));
         }
       }
     }
   }
}

/**
 * Concatenate many ropes.
 *
 * If the ropes are balanced initially and have the same height, the resulting
 * rope remains balanced. However, this function does not take any further
 * measure to ensure that the result is balanced.
 */
fn concat(v: ~[rope]) -> rope {
    //Copy `v` into a mut vector
    let mut len = vec::len(v);
    if len == 0u { return node::empty; }
    let ropes = vec::to_mut(vec::from_elem(len, v[0]));
    for uint::range(1u, len) |i| {
       ropes[i] = v[i];
    }

    //Merge progresively
    while len > 1u {
        for uint::range(0u, len/2u) |i| {
            ropes[i] = append_rope(ropes[2u*i], ropes[2u*i+1u]);
        }
        if len%2u != 0u {
            ropes[len/2u] = ropes[len - 1u];
            len = len/2u + 1u;
        } else {
            len = len/2u;
        }
    }

    //Return final rope
    return ropes[0];
}


/*
Section: Keeping ropes healthy
 */


/**
 * Balance a rope.
 *
 * # Return value
 *
 * A copy of the rope in which small nodes have been grouped in memory,
 * and with a reduced height.
 *
 * If you perform numerous rope concatenations, it is generally a good idea
 * to rebalance your rope at some point, before using it for other purposes.
 */
fn bal(rope:rope) -> rope {
    match (rope) {
      node::empty => return rope,
      node::content(x) => match (node::bal(x)) {
        option::None    => rope,
        option::Some(y) => node::content(y)
      }
    }
}

/*
Section: Transforming ropes
 */


/**
 * Extract a subrope from a rope.
 *
 * # Performance note
 *
 * * on a balanced rope, this operation takes algorithmic time;
 * * this operation does not involve any copying
 *
 * # Safety note
 *
 * * this function fails if char_offset/char_len do not represent
 *   valid positions in rope
 */
fn sub_chars(rope: rope, char_offset: uint, char_len: uint) -> rope {
    if char_len == 0u { return node::empty; }
    match (rope) {
      node::empty => fail,
      node::content(node) => if char_len > node::char_len(node) {
        fail
      } else {
        return node::content(node::sub_chars(node, char_offset, char_len))
      }
    }
}

/**
 * Extract a subrope from a rope.
 *
 * # Performance note
 *
 * * on a balanced rope, this operation takes algorithmic time;
 * * this operation does not involve any copying
 *
 * # Safety note
 *
 * * this function fails if byte_offset/byte_len do not represent
 *   valid positions in rope
 */
fn sub_bytes(rope: rope, byte_offset: uint, byte_len: uint) -> rope {
    if byte_len == 0u { return node::empty; }
    match (rope) {
      node::empty => fail,
      node::content(node) =>if byte_len > node::byte_len(node) {
        fail
      } else {
        return node::content(node::sub_bytes(node, byte_offset, byte_len))
      }
    }
}

/*
Section: Comparing ropes
 */

/**
 * Compare two ropes by Unicode lexicographical order.
 *
 * This function compares only the contents of the rope, not their structure.
 *
 * # Return value
 *
 * A negative value if `left < right`, 0 if eq(left, right) or a positive
 * value if `left > right`
 */
fn cmp(left: rope, right: rope) -> int {
    match ((left, right)) {
      (node::empty, node::empty) => return 0,
      (node::empty, _)     => return -1,
      (_, node::empty)     => return  1,
      (node::content(a), node::content(b)) => {
        return node::cmp(a, b);
      }
    }
}

/**
 * Returns `true` if both ropes have the same content (regardless of
 * their structure), `false` otherwise
 */
fn eq(left: rope, right: rope) -> bool {
    return cmp(left, right) == 0;
}

/**
 * # Arguments
 *
 * * left - an arbitrary rope
 * * right - an arbitrary rope
 *
 * # Return value
 *
 * `true` if `left <= right` in lexicographical order (regardless of their
 * structure), `false` otherwise
 */
fn le(left: rope, right: rope) -> bool {
    return cmp(left, right) <= 0;
}

/**
 * # Arguments
 *
 * * left - an arbitrary rope
 * * right - an arbitrary rope
 *
 * # Return value
 *
 * `true` if `left < right` in lexicographical order (regardless of their
 * structure), `false` otherwise
 */
fn lt(left: rope, right: rope) -> bool {
    return cmp(left, right) < 0;
}

/**
 * # Arguments
 *
 * * left - an arbitrary rope
 * * right - an arbitrary rope
 *
 * # Return value
 *
 *  `true` if `left >= right` in lexicographical order (regardless of their
 * structure), `false` otherwise
 */
fn ge(left: rope, right: rope) -> bool {
    return cmp(left, right) >= 0;
}

/**
 * # Arguments
 *
 * * left - an arbitrary rope
 * * right - an arbitrary rope
 *
 * # Return value
 *
 * `true` if `left > right` in lexicographical order (regardless of their
 * structure), `false` otherwise
 */
fn gt(left: rope, right: rope) -> bool {
    return cmp(left, right) > 0;
}

/*
Section: Iterating
 */

/**
 * Loop through a rope, char by char
 *
 * While other mechanisms are available, this is generally the best manner
 * of looping through the contents of a rope char by char. If you prefer a
 * loop that iterates through the contents string by string (e.g. to print
 * the contents of the rope or output it to the system), however,
 * you should rather use `traverse_components`.
 *
 * # Arguments
 *
 * * rope - A rope to traverse. It may be empty.
 * * it - A block to execute with each consecutive character of the rope.
 *        Return `true` to continue, `false` to stop.
 *
 * # Return value
 *
 * `true` If execution proceeded correctly, `false` if it was interrupted,
 * that is if `it` returned `false` at any point.
 */
fn loop_chars(rope: rope, it: fn(char) -> bool) -> bool {
   match (rope) {
      node::empty => return true,
      node::content(x) => return node::loop_chars(x, it)
   }
}

/**
 * Loop through a rope, char by char, until the end.
 *
 * # Arguments
 * * rope - A rope to traverse. It may be empty
 * * it - A block to execute with each consecutive character of the rope.
 */
fn iter_chars(rope: rope, it: fn(char)) {
    do loop_chars(rope) |x| {
        it(x);
        true
    };
}

/**
 * Loop through a rope, string by string
 *
 * While other mechanisms are available, this is generally the best manner of
 * looping through the contents of a rope string by string, which may be
 * useful e.g. to print strings as you see them (without having to copy their
 * contents into a new string), to send them to then network, to write them to
 * a file, etc.. If you prefer a loop that iterates through the contents
 * char by char (e.g. to search for a char), however, you should rather
 * use `traverse`.
 *
 * # Arguments
 *
 * * rope - A rope to traverse. It may be empty
 * * it - A block to execute with each consecutive string component of the
 *        rope. Return `true` to continue, `false` to stop
 *
 * # Return value
 *
 * `true` If execution proceeded correctly, `false` if it was interrupted,
 * that is if `it` returned `false` at any point.
 */
fn loop_leaves(rope: rope, it: fn(node::leaf) -> bool) -> bool{
   match (rope) {
      node::empty => return true,
      node::content(x) => return node::loop_leaves(x, it)
   }
}

mod iterator {
    mod leaf {
        fn start(rope: rope) -> node::leaf_iterator::t {
            match (rope) {
              node::empty      => return node::leaf_iterator::empty(),
              node::content(x) => return node::leaf_iterator::start(x)
            }
        }
        fn next(it: &node::leaf_iterator::t) -> Option<node::leaf> {
            return node::leaf_iterator::next(it);
        }
    }
    mod char {
        fn start(rope: rope) -> node::char_iterator::t {
            match (rope) {
              node::empty      => return node::char_iterator::empty(),
              node::content(x) => return node::char_iterator::start(x)
            }
        }
        fn next(it: &node::char_iterator::t) -> Option<char> {
            return node::char_iterator::next(it)
        }
    }
}

/*
 Section: Rope properties
 */

/**
 * Returns the height of the rope.
 *
 * The height of the rope is a bound on the number of operations which
 * must be performed during a character access before finding the leaf in
 * which a character is contained.
 *
 * # Performance note
 *
 * Constant time.
 */
fn height(rope: rope) -> uint {
   match (rope) {
      node::empty      => return 0u,
      node::content(x) => return node::height(x)
   }
}



/**
 * The number of character in the rope
 *
 * # Performance note
 *
 * Constant time.
 */
pure fn char_len(rope: rope) -> uint {
   match (rope) {
     node::empty            => return 0u,
     node::content(x)       => return node::char_len(x)
   }
}

/**
 * The number of bytes in the rope
 *
 * # Performance note
 *
 * Constant time.
 */
pure fn byte_len(rope: rope) -> uint {
   match (rope) {
     node::empty            => return 0u,
     node::content(x)       => return node::byte_len(x)
   }
}

/**
 * The character at position `pos`
 *
 * # Arguments
 *
 * * pos - A position in the rope
 *
 * # Safety notes
 *
 * The function will fail if `pos` is not a valid position in the rope.
 *
 * # Performance note
 *
 * This function executes in a time proportional to the height of the
 * rope + the (bounded) length of the largest leaf.
 */
fn char_at(rope: rope, pos: uint) -> char {
   match (rope) {
      node::empty => fail,
      node::content(x) => return node::char_at(x, pos)
   }
}


/*
 Section: Implementation
*/
mod node {

    /// Implementation of type `rope`
    enum root {
        /// An empty rope
        empty,
        /// A non-empty rope
        content(@node),
    }

    /**
     * A text component in a rope.
     *
     * This is actually a slice in a rope, so as to ensure maximal sharing.
     *
     * # Fields
     *
     * * byte_offset = The number of bytes skippen in `content`
     * * byte_len - The number of bytes of `content` to use
     * * char_len - The number of chars in the leaf.
     * * content - Contents of the leaf.
     *
     *     Note that we can have `char_len < str::char_len(content)`, if
     *     this leaf is only a subset of the string. Also note that the
     *     string can be shared between several ropes, e.g. for indexing
     *     purposes.
     */
    type leaf = {
        byte_offset: uint,
        byte_len:    uint,
        char_len:   uint,
        content:    @~str
    };

    /**
     * A node obtained from the concatenation of two other nodes
     *
     * # Fields
     *
     * * left - The node containing the beginning of the text.
     * * right - The node containing the end of the text.
     * * char_len - The number of chars contained in all leaves of this node.
     * * byte_len - The number of bytes in the subrope.
     *
     *     Used to pre-allocate the correct amount of storage for
     *     serialization.
     *
     * * height - Height of the subrope.
     *
     *     Used for rebalancing and to allocate stacks for traversals.
     */
    type concat = {
        //FIXME (#2744): Perhaps a `vec` instead of `left`/`right`
        left:     @node,
        right:    @node,
        char_len: uint,
        byte_len: uint,
        height:   uint
    };

    enum node {
        /// A leaf consisting in a `str`
        leaf(leaf),
        /// The concatenation of two ropes
        concat(concat),
    }

    /**
     * The maximal number of chars that _should_ be permitted in a single node
     *
     * This is not a strict value
     */
    const hint_max_leaf_char_len: uint = 256u;

    /**
     * The maximal height that _should_ be permitted in a tree.
     *
     * This is not a strict value
     */
    const hint_max_node_height:   uint = 16u;

    /**
     * Adopt a string as a node.
     *
     * If the string is longer than `max_leaf_char_len`, it is
     * logically split between as many leaves as necessary. Regardless,
     * the string itself is not copied.
     *
     * Performance note: The complexity of this function is linear in
     * the length of `str`.
     */
    fn of_str(str: @~str) -> @node {
        return of_substr(str, 0u, str::len(*str));
    }

    /**
     * Adopt a slice of a string as a node.
     *
     * If the slice is longer than `max_leaf_char_len`, it is logically split
     * between as many leaves as necessary. Regardless, the string itself
     * is not copied
     *
     * # Arguments
     *
     * * byte_start - The byte offset where the slice of `str` starts.
     * * byte_len   - The number of bytes from `str` to use.
     *
     * # Safety note
     *
     * Behavior is undefined if `byte_start` or `byte_len` do not represent
     * valid positions in `str`
     */
    fn of_substr(str: @~str, byte_start: uint, byte_len: uint) -> @node {
        return of_substr_unsafer(str, byte_start, byte_len,
                              str::count_chars(*str, byte_start, byte_len));
    }

    /**
     * Adopt a slice of a string as a node.
     *
     * If the slice is longer than `max_leaf_char_len`, it is logically split
     * between as many leaves as necessary. Regardless, the string itself
     * is not copied
     *
     * # Arguments
     *
     * * byte_start - The byte offset where the slice of `str` starts.
     * * byte_len - The number of bytes from `str` to use.
     * * char_len - The number of chars in `str` in the interval
     *              [byte_start, byte_start+byte_len)
     *
     * # Safety notes
     *
     * * Behavior is undefined if `byte_start` or `byte_len` do not represent
     *   valid positions in `str`
     * * Behavior is undefined if `char_len` does not accurately represent the
     *   number of chars between byte_start and byte_start+byte_len
     */
    fn of_substr_unsafer(str: @~str, byte_start: uint, byte_len: uint,
                          char_len: uint) -> @node {
        assert(byte_start + byte_len <= str::len(*str));
        let candidate = @leaf({
                byte_offset: byte_start,
                byte_len:    byte_len,
                char_len:    char_len,
                content:     str});
        if char_len <= hint_max_leaf_char_len {
            return candidate;
        } else {
            //Firstly, split `str` in slices of hint_max_leaf_char_len
            let mut leaves = uint::div_ceil(char_len, hint_max_leaf_char_len);
            //Number of leaves
            let nodes  = vec::to_mut(vec::from_elem(leaves, candidate));

            let mut i = 0u;
            let mut offset = byte_start;
            let first_leaf_char_len =
                if char_len%hint_max_leaf_char_len == 0u {
                  hint_max_leaf_char_len
                } else {
                char_len%hint_max_leaf_char_len
               };
            while i < leaves {
                let chunk_char_len: uint =
                    if i == 0u  { first_leaf_char_len }
                    else { hint_max_leaf_char_len };
                let chunk_byte_len =
                    str::count_bytes(*str, offset, chunk_char_len);
                nodes[i] = @leaf({
                    byte_offset: offset,
                    byte_len:    chunk_byte_len,
                    char_len:    chunk_char_len,
                    content:     str
                });

                offset += chunk_byte_len;
                i      += 1u;
            }

            //Then, build a tree from these slices by collapsing them
            while leaves > 1u {
                i = 0u;
                while i < leaves - 1u {//Concat nodes 0 with 1, 2 with 3 etc.
                    nodes[i/2u] = concat2(nodes[i], nodes[i + 1u]);
                    i += 2u;
                }
                if i == leaves - 1u {
                    //And don't forget the last node if it is in even position
                    nodes[i/2u] = nodes[i];
                }
                leaves = uint::div_ceil(leaves, 2u);
            }
            return nodes[0u];
        }
    }

    pure fn byte_len(node: @node) -> uint {
        //FIXME (#2744): Could we do this without the pattern-matching?
        match (*node) {
          leaf(y)   => return y.byte_len,
          concat(y) => return y.byte_len
        }
    }

    pure fn char_len(node: @node) -> uint {
        match (*node) {
          leaf(y)   => return y.char_len,
          concat(y) => return y.char_len
        }
    }

    /**
     * Concatenate a forest of nodes into one tree.
     *
     * # Arguments
     *
     * * forest - The forest. This vector is progressively rewritten during
     *            execution and should be discarded as meaningless afterwards.
     */
    fn tree_from_forest_destructive(forest: &[mut @node]) -> @node {
        let mut i;
        let mut len = vec::len(forest);
        while len > 1u {
            i = 0u;
            while i < len - 1u {//Concat nodes 0 with 1, 2 with 3 etc.
                let mut left  = forest[i];
                let mut right = forest[i+1u];
                let left_len = char_len(left);
                let right_len= char_len(right);
                let mut left_height= height(left);
                let mut right_height=height(right);
                if left_len + right_len > hint_max_leaf_char_len {
                    if left_len <= hint_max_leaf_char_len {
                        left = flatten(left);
                        left_height = height(left);
                    }
                    if right_len <= hint_max_leaf_char_len {
                        right = flatten(right);
                        right_height = height(right);
                    }
                }
                if left_height >= hint_max_node_height {
                    left = of_substr_unsafer(@serialize_node(left),
                                             0u,byte_len(left),
                                             left_len);
                }
                if right_height >= hint_max_node_height {
                    right = of_substr_unsafer(@serialize_node(right),
                                             0u,byte_len(right),
                                             right_len);
                }
                forest[i/2u] = concat2(left, right);
                i += 2u;
            }
            if i == len - 1u {
                //And don't forget the last node if it is in even position
                forest[i/2u] = forest[i];
            }
            len = uint::div_ceil(len, 2u);
        }
        return forest[0];
    }

    fn serialize_node(node: @node) -> ~str unsafe {
        let mut buf = vec::to_mut(vec::from_elem(byte_len(node), 0u8));
        let mut offset = 0u;//Current position in the buffer
        let it = leaf_iterator::start(node);
        loop {
            match (leaf_iterator::next(&it)) {
              option::None => break,
              option::Some(x) => {
                //FIXME (#2744): Replace with memcpy or something similar
                let mut local_buf: ~[u8] =
                    unsafe::reinterpret_cast(*x.content);
                let mut i = x.byte_offset;
                while i < x.byte_len {
                    buf[offset] = local_buf[i];
                    offset += 1u;
                    i      += 1u;
                }
                unsafe::forget(local_buf);
              }
            }
        }
        return unsafe::transmute(buf);
    }

    /**
     * Replace a subtree by a single leaf with the same contents.
     *
     * * Performance note
     *
     * This function executes in linear time.
     */
    fn flatten(node: @node) -> @node unsafe {
        match (*node) {
          leaf(_) => return node,
          concat(x) => {
            return @leaf({
                byte_offset: 0u,
                byte_len:    x.byte_len,
                char_len:    x.char_len,
                content:     @serialize_node(node)
            })
          }
        }
    }

    /**
     * Balance a node.
     *
     * # Algorithm
     *
     * * if the node height is smaller than `hint_max_node_height`, do nothing
     * * otherwise, gather all leaves as a forest, rebuild a balanced node,
     *   concatenating small leaves along the way
     *
     * # Return value
     *
     * * `option::None` if no transformation happened
     * * `option::some(x)` otherwise, in which case `x` has the same contents
     *    as `node` bot lower height and/or fragmentation.
     */
    fn bal(node: @node) -> Option<@node> {
        if height(node) < hint_max_node_height { return option::None; }
        //1. Gather all leaves as a forest
        let mut forest = ~[mut];
        let it = leaf_iterator::start(node);
        loop {
            match (leaf_iterator::next(&it)) {
              option::None    => break,
              option::Some(x) => vec::push(forest, @leaf(x))
            }
        }
        //2. Rebuild tree from forest
        let root = @*tree_from_forest_destructive(forest);
        return option::Some(root);

    }

    /**
     * Compute the subnode of a node.
     *
     * # Arguments
     *
     * * node        - A node
     * * byte_offset - A byte offset in `node`
     * * byte_len    - The number of bytes to return
     *
     * # Performance notes
     *
     * * this function performs no copying;
     * * this function executes in a time proportional to the height of `node`
     *
     * # Safety notes
     *
     * This function fails if `byte_offset` or `byte_len` do not represent
     * valid positions in `node`.
     */
    fn sub_bytes(node: @node, byte_offset: uint, byte_len: uint) -> @node {
        let mut node        = node;
        let mut byte_offset = byte_offset;
        loop {
            if byte_offset == 0u && byte_len == node::byte_len(node) {
                return node;
            }
            match (*node) {
              node::leaf(x) => {
                let char_len =
                    str::count_chars(*x.content, byte_offset, byte_len);
                return @leaf({byte_offset: byte_offset,
                                byte_len:    byte_len,
                                char_len:    char_len,
                                content:     x.content});
              }
              node::concat(x) => {
                let left_len: uint = node::byte_len(x.left);
                if byte_offset <= left_len {
                    if byte_offset + byte_len <= left_len {
                        //Case 1: Everything fits in x.left, tail-call
                        node = x.left;
                    } else {
                        //Case 2: A (non-empty, possibly full) suffix
                        //of x.left and a (non-empty, possibly full) prefix
                        //of x.right
                        let left_result  =
                            sub_bytes(x.left, byte_offset, left_len);
                        let right_result =
                            sub_bytes(x.right, 0u, left_len - byte_offset);
                        return concat2(left_result, right_result);
                    }
                } else {
                    //Case 3: Everything fits in x.right
                    byte_offset -= left_len;
                    node = x.right;
                }
              }
            }
        };
    }

    /**
     * Compute the subnode of a node.
     *
     * # Arguments
     *
     * * node        - A node
     * * char_offset - A char offset in `node`
     * * char_len    - The number of chars to return
     *
     * # Performance notes
     *
     * * this function performs no copying;
     * * this function executes in a time proportional to the height of `node`
     *
     * # Safety notes
     *
     * This function fails if `char_offset` or `char_len` do not represent
     * valid positions in `node`.
     */
    fn sub_chars(node: @node, char_offset: uint, char_len: uint) -> @node {
        let mut node        = node;
        let mut char_offset = char_offset;
        loop {
            match (*node) {
              node::leaf(x) => {
                if char_offset == 0u && char_len == x.char_len {
                    return node;
                }
                let byte_offset =
                    str::count_bytes(*x.content, 0u, char_offset);
                let byte_len    =
                    str::count_bytes(*x.content, byte_offset, char_len);
                return @leaf({byte_offset: byte_offset,
                           byte_len:    byte_len,
                           char_len:    char_len,
                           content:     x.content});
              }
              node::concat(x) => {
                if char_offset == 0u && char_len == x.char_len {return node;}
                let left_len : uint = node::char_len(x.left);
                if char_offset <= left_len {
                    if char_offset + char_len <= left_len {
                        //Case 1: Everything fits in x.left, tail call
                        node        = x.left;
                    } else {
                        //Case 2: A (non-empty, possibly full) suffix
                        //of x.left and a (non-empty, possibly full) prefix
                        //of x.right
                        let left_result  =
                            sub_chars(x.left, char_offset, left_len);
                        let right_result =
                            sub_chars(x.right, 0u, left_len - char_offset);
                        return concat2(left_result, right_result);
                    }
                } else {
                    //Case 3: Everything fits in x.right, tail call
                    node = x.right;
                    char_offset -= left_len;
                }
              }
            }
        };
    }

    fn concat2(left: @node, right: @node) -> @node {
        return @concat({left    : left,
                     right   : right,
             char_len: char_len(left) + char_len(right),
                     byte_len: byte_len(left) + byte_len(right),
             height: uint::max(height(left), height(right)) + 1u
                    })
    }

    fn height(node: @node) -> uint {
        match (*node) {
          leaf(_)   => return 0u,
          concat(x) => return x.height
        }
    }

    fn cmp(a: @node, b: @node) -> int {
        let ita = char_iterator::start(a);
        let itb = char_iterator::start(b);
        let mut result = 0;
        while result == 0 {
            match ((char_iterator::next(&ita), char_iterator::next(&itb))) {
              (option::None, option::None) => break,
              (option::Some(chara), option::Some(charb)) => {
                result = char::cmp(chara, charb);
              }
              (option::Some(_), _) => {
                result = 1;
              }
              (_, option::Some(_)) => {
                result = -1;
              }
            }
        }
        return result;
    }

    fn loop_chars(node: @node, it: fn(char) -> bool) -> bool {
        return loop_leaves(node,|leaf| {
            str::all_between(*leaf.content,
                             leaf.byte_offset,
                             leaf.byte_len, it)
        });
    }

    /**
     * Loop through a node, leaf by leaf
     *
     * # Arguments
     *
     * * rope - A node to traverse.
     * * it - A block to execute with each consecutive leaf of the node.
     *        Return `true` to continue, `false` to stop
     *
     * # Arguments
     *
     * `true` If execution proceeded correctly, `false` if it was interrupted,
     * that is if `it` returned `false` at any point.
     */
    fn loop_leaves(node: @node, it: fn(leaf) -> bool) -> bool{
        let mut current = node;
        loop {
            match (*current) {
              leaf(x) => return it(x),
              concat(x) => if loop_leaves(x.left, it) { //non tail call
                current = x.right;       //tail call
              } else {
                return false;
              }
            }
        };
    }

    /**
     * # Arguments
     *
     * * pos - A position in the rope
     *
     * # Return value
     *
     * The character at position `pos`
     *
     * # Safety notes
     *
     * The function will fail if `pos` is not a valid position in the rope.
     *
     * Performance note: This function executes in a time
     * proportional to the height of the rope + the (bounded)
     * length of the largest leaf.
     */
    fn char_at(node: @node, pos: uint) -> char {
        let mut node    = node;
        let mut pos     = pos;
        loop {
            match *node {
              leaf(x) => return str::char_at(*x.content, pos),
              concat({left, right, _}) => {
                let left_len = char_len(left);
                node = if left_len > pos { left }
                       else { pos -= left_len; right };
              }
            }
        };
    }

    mod leaf_iterator {
        type t = {
            stack:            ~[mut @node],
            mut stackpos: int
        };

        fn empty() -> t {
            let stack : ~[mut @node] = ~[mut];
            return {stack: stack, mut stackpos: -1}
        }

        fn start(node: @node) -> t {
            let stack = vec::to_mut(vec::from_elem(height(node)+1u, node));
            return {
                stack:             stack,
                mut stackpos:  0
            }
        }

        fn next(it: &t) -> Option<leaf> {
            if it.stackpos < 0 { return option::None; }
            loop {
                let current = it.stack[it.stackpos];
                it.stackpos -= 1;
                match (*current) {
                  concat(x) => {
                    it.stackpos += 1;
                    it.stack[it.stackpos] = x.right;
                    it.stackpos += 1;
                    it.stack[it.stackpos] = x.left;
                  }
                  leaf(x) => return option::Some(x)
                }
            };
        }
    }

    mod char_iterator {
        type t = {
            leaf_iterator: leaf_iterator::t,
            mut leaf:  Option<leaf>,
            mut leaf_byte_pos: uint
        };

        fn start(node: @node) -> t {
            return {
                leaf_iterator: leaf_iterator::start(node),
                mut leaf:          option::None,
                mut leaf_byte_pos: 0u
            }
        }

        fn empty() -> t {
            return {
                leaf_iterator: leaf_iterator::empty(),
                mut leaf:  option::None,
                mut leaf_byte_pos: 0u
            }
        }

        fn next(it: &t) -> Option<char> {
            loop {
                match (get_current_or_next_leaf(it)) {
                  option::None => return option::None,
                  option::Some(_) => {
                    let next_char = get_next_char_in_leaf(it);
                    match (next_char) {
                      option::None => again,
                      option::Some(_) => return next_char
                    }
                  }
                }
            };
        }

        fn get_current_or_next_leaf(it: &t) -> Option<leaf> {
            match ((*it).leaf) {
              option::Some(_) => return (*it).leaf,
              option::None => {
                let next = leaf_iterator::next(&((*it).leaf_iterator));
                match (next) {
                  option::None => return option::None,
                  option::Some(_) => {
                    (*it).leaf          = next;
                    (*it).leaf_byte_pos = 0u;
                    return next;
                  }
                }
              }
            }
        }

        fn get_next_char_in_leaf(it: &t) -> Option<char> {
            match copy (*it).leaf {
              option::None => return option::None,
              option::Some(aleaf) => {
                if (*it).leaf_byte_pos >= aleaf.byte_len {
                    //We are actually past the end of the leaf
                    (*it).leaf = option::None;
                    return option::None
                } else {
                    let {ch, next} =
                        str::char_range_at(*aleaf.content,
                                     (*it).leaf_byte_pos + aleaf.byte_offset);
                    (*it).leaf_byte_pos = next - aleaf.byte_offset;
                    return option::Some(ch)
                }
              }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    //Utility function, used for sanity check
    fn rope_to_string(r: rope) -> ~str {
        match (r) {
          node::empty => return ~"",
          node::content(x) => {
            let str = @mut ~"";
            fn aux(str: @mut ~str, node: @node::node) unsafe {
                match (*node) {
                  node::leaf(x) => {
                    *str += str::slice(
                        *x.content, x.byte_offset,
                        x.byte_offset + x.byte_len);
                  }
                  node::concat(x) => {
                    aux(str, x.left);
                    aux(str, x.right);
                  }
                }
            }
            aux(str, x);
            return *str
          }
        }
    }


    #[test]
    fn trivial() {
        assert char_len(empty()) == 0u;
        assert byte_len(empty()) == 0u;
    }

    #[test]
    fn of_string1() {
        let sample = @~"0123456789ABCDE";
        let r      = of_str(sample);

        assert char_len(r) == str::char_len(*sample);
        assert rope_to_string(r) == *sample;
    }

    #[test]
    fn of_string2() {
        let buf = @ mut ~"1234567890";
        let mut i = 0;
        while i < 10 { *buf = *buf + *buf; i+=1;}
        let sample = @*buf;
        let r      = of_str(sample);
        assert char_len(r) == str::char_len(*sample);
        assert rope_to_string(r) == *sample;

        let mut string_iter = 0u;
        let string_len  = str::len(*sample);
        let rope_iter   = iterator::char::start(r);
        let mut equal   = true;
        while equal {
            match (node::char_iterator::next(&rope_iter)) {
              option::None => {
                if string_iter < string_len {
                    equal = false;
                } break; }
              option::Some(c) => {
                let {ch, next} = str::char_range_at(*sample, string_iter);
                string_iter = next;
                if ch != c { equal = false; break; }
              }
            }
        }

        assert equal;
    }

    #[test]
    fn iter1() {
        let buf = @ mut ~"1234567890";
        let mut i = 0;
        while i < 10 { *buf = *buf + *buf; i+=1;}
        let sample = @*buf;
        let r      = of_str(sample);

        let mut len = 0u;
        let it  = iterator::char::start(r);
        loop {
            match (node::char_iterator::next(&it)) {
              option::None => break,
              option::Some(_) => len += 1u
            }
        }

        assert len == str::char_len(*sample);
    }

    #[test]
    fn bal1() {
        let init = @~"1234567890";
        let buf  = @mut * init;
        let mut i = 0;
        while i < 8 { *buf = *buf + *buf; i+=1;}
        let sample = @*buf;
        let r1     = of_str(sample);
        let mut r2 = of_str(init);
        i = 0;
        while i < 8 { r2 = append_rope(r2, r2); i+= 1;}


        assert eq(r1, r2);
        let r3 = bal(r2);
        assert char_len(r1) == char_len(r3);

        assert eq(r1, r3);
    }

    #[test]
    #[ignore]
    fn char_at1() {
        //Generate a large rope
        let mut r = of_str(@~"123456789");
        for uint::range(0u, 10u) |_i| {
            r = append_rope(r, r);
        }

        //Copy it in the slowest possible way
        let mut r2 = empty();
        for uint::range(0u, char_len(r)) |i| {
            r2 = append_char(r2, char_at(r, i));
        }
        assert eq(r, r2);

        let mut r3 = empty();
        for uint::range(0u, char_len(r)) |i| {
            r3 = prepend_char(r3, char_at(r, char_len(r) - i - 1u));
        }
        assert eq(r, r3);

        //Additional sanity checks
        let balr = bal(r);
        let bal2 = bal(r2);
        let bal3 = bal(r3);
        assert eq(r, balr);
        assert eq(r, bal2);
        assert eq(r, bal3);
        assert eq(r2, r3);
        assert eq(bal2, bal3);
    }

    #[test]
    fn concat1() {
        //Generate a reasonable rope
        let chunk = of_str(@~"123456789");
        let mut r = empty();
        for uint::range(0u, 10u) |_i| {
            r = append_rope(r, chunk);
        }

        //Same rope, obtained with rope::concat
        let r2 = concat(vec::from_elem(10u, chunk));

        assert eq(r, r2);
    }
}
