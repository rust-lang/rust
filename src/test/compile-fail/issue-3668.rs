struct P { child: Option<@mut P> }
trait PTrait {
   fn getChildOption() -> Option<@P>;
}

impl P: PTrait {
   fn getChildOption() -> Option<@P> {
       const childVal: @P = self.child.get(); //~ ERROR attempt to use a non-constant value in a constant
       fail;
   }
}

fn main() {}
