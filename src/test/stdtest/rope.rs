import std::str;
import std::rope::*;
import std::option;
import std::uint;
import std::vec;

//Utility function, used for sanity check
fn rope_to_string(r: rope) -> str {
    alt(r) {
      node::empty. { ret "" }
      node::content(x) {
        let str = @mutable "";
        fn aux(str: @mutable str, node: @node::node) {
            alt(*node) {
              node::leaf(x) {
                *str += str::substr(*x.content, x.byte_offset, x.byte_len);
              }
              node::concat(x) {
                aux(str, x.left);
                aux(str, x.right);
              }
            }
        }
        aux(str, x);
        ret *str
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
    let sample = @"0123456789ABCDE";
    let r      = of_str(sample);

    assert char_len(r) == str::char_len(*sample);
    assert rope_to_string(r) == *sample;
}

#[test]
fn of_string2() {
    let buf = @ mutable "1234567890";
    let i = 0;
    while i < 10 { *buf = *buf + *buf; i+=1;}
    let sample = @*buf;
    let r      = of_str(sample);
    assert char_len(r) == str::char_len(*sample);
    assert rope_to_string(r) == *sample;

    let string_iter = 0u;
    let string_len  = str::byte_len(*sample);
    let rope_iter   = iterator::char::start(r);
    let equal       = true;
    let pos         = 0u;
    while equal {
        alt(node::char_iterator::next(rope_iter)) {
          option::none. {
            if string_iter < string_len {
                equal = false;
            } break; }
          option::some(c) {
            let {ch, next} = str::char_range_at(*sample, string_iter);
            string_iter = next;
            if ch != c { equal = false; break; }
          }
        }
        pos += 1u;
    }

    assert equal;
}

#[test]
fn iter1() {
    let buf = @ mutable "1234567890";
    let i = 0;
    while i < 10 { *buf = *buf + *buf; i+=1;}
    let sample = @*buf;
    let r      = of_str(sample);

    let len = 0u;
    let it  = iterator::char::start(r);
    while true {
        alt(node::char_iterator::next(it)) {
          option::none. { break; }
          option::some(_) { len += 1u; }
        }
    }

    assert len == str::char_len(*sample);
}

#[test]
fn bal1() {
    let init = @ "1234567890";
    let buf  = @ mutable * init;
    let i = 0;
    while i < 16 { *buf = *buf + *buf; i+=1;}
    let sample = @*buf;
    let r1     = of_str(sample);
    let r2     = of_str(init);
    i = 0;
    while i < 16 { r2 = append_rope(r2, r2); i+= 1;}


    assert eq(r1, r2);
    let r3 = bal(r2);
    assert char_len(r1) == char_len(r3);

    assert eq(r1, r3);
}

#[test]
fn char_at1() {
    //Generate a large rope
    let r = of_str(@ "123456789");
    uint::range(0u, 10u){|_i|
        r = append_rope(r, r);
    }

    //Copy it in the slowest possible way
    let r2 = empty();
    uint::range(0u, char_len(r)){|i|
        r2 = append_char(r2, char_at(r, i));
    }
    assert eq(r, r2);

    let r3 = empty();
    uint::range(0u, char_len(r)){|i|
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
    let chunk = of_str(@ "123456789");
    let r = empty();
    uint::range(0u, 10u){|_i|
        r = append_rope(r, chunk);
    }

    //Same rope, obtained with rope::concat
    let r2 = concat(vec::init_elt(chunk, 10u));

    assert eq(r, r2);
}