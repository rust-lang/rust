// Closures

fn main() {
    let square = (|i: i32| i * i);

    let commented = |// first
                     a, // argument
                     // second
                     b: WithType, // argument
                     // ignored
                     _| (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbb);

    let commented = |// first
                     a, // argument
                     // second
                     b: WithType, // argument
                     // ignored
                     _| {
                        (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
                         bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)
                    };

    let block_body = move |xxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
                           ref yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy| {
                              xxxxxxxxxxxxxxxxxxxxxxxxxxxxx + yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
                          };

    let loooooooooooooong_name = |field| {
             // TODO(#27): format comments.
                                     if field.node.attrs.len() > 0 {
                                         field.node.attrs[0].span.lo
                                     } else {
                                         field.span.lo
                                     }
                                 };

    let block_me = |field| {
                       if true_story() {
                           1
                       } else {
                           2
                       }
                   };

    let unblock_me = |trivial| closure();

    let empty = |arg| {};
}
