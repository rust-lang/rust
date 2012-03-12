use std;
import std::prettyprint::serializer;
import std::io;

#[auto_serialize]
enum expr {
    val(uint),
    plus(@expr, @expr),
    minus(@expr, @expr)
}

fn main() {
    let ex = @plus(@minus(@val(3u), @val(10u)),
                   @plus(@val(22u), @val(5u)));
    let s = io::with_str_writer {|w| expr::serialize(w, *ex)};
    #debug["s == %?", s];
    assert s == "plus(@minus(@val(3u), @val(10u)), \
                 @plus(@val(22u), @val(5u)))";
}