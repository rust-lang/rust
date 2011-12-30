use std;

fn asSendfn( f : sendfn()->uint ) -> uint {
   ret f();
}

fn asLambda( f : lambda()->uint ) -> uint {
   ret f();
}

fn asBlock( f : block()->uint ) -> uint {
   ret f();
}

fn main() {
   let x = asSendfn({|| 22u});
   assert(x == 22u);
   let x = asLambda({|| 22u});
   assert(x == 22u);
   let x = asBlock({|| 22u});
   assert(x == 22u);
}
