extern mod std;

fn asSendfn( f : fn~()->uint ) -> uint {
   return f();
}

fn asLambda( f : fn@()->uint ) -> uint {
   return f();
}

fn asBlock( f : fn&()->uint ) -> uint {
   return f();
}

fn asAny( f : fn()->uint ) -> uint {
   return f();
}

fn main() {
   let x = asSendfn(|| 22u);
   assert(x == 22u);
   let x = asLambda(|| 22u);
   assert(x == 22u);
   let x = asBlock(|| 22u);
   assert(x == 22u);
}
