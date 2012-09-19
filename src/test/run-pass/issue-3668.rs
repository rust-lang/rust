// xfail-test
struct P { child: Option<@mut P> }
trait PTrait {
   fn getChildOption() -> Option<@P>;
}

impl P: PTrait {
   fn getChildOption() -> Option<@P> {
       const childVal: @P = self.child.get();
       fail;
   }
}

fn main() {}
