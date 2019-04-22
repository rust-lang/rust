// This is a test adapted from a minimization of the code from
// rust-lang/rust#52934, where an accidental disabling of
// two-phase-borrows (in the initial 2018 edition integration) broke
// Clippy, but the scenarios where it was breaking were subtle enough
// that we decided it warranted its own unit test, and pnkfelix
// decided to use that test as an opportunity to illustrate the cases.

#[derive(Copy, Clone)]
struct BodyId;
enum Expr { Closure(BodyId), Others }
struct Body { value: Expr }

struct Map { body: Body, }
impl Map { fn body(&self, _: BodyId) -> &Body { unimplemented!() } }

struct SpanlessHash<'a> { cx: &'a Map, cx_mut: &'a mut Map }

impl <'a> SpanlessHash<'a> {
    fn demo(&mut self) {
        let _mut_borrow = &mut *self;
        let _access = self.cx;
        //~^ ERROR cannot use `self.cx` because it was mutably borrowed [E0503]
        _mut_borrow;
    }

    fn hash_expr(&mut self, e: &Expr) {
        match *e {
            Expr::Closure(eid) => {
                // Accepted by AST-borrowck for erroneous reasons
                // (rust-lang/rust#38899).
                //
                // Not okay without two-phase borrows: the implicit
                // `&mut self` of the receiver is evaluated first, and
                // that conflicts with the `self.cx`` access during
                // argument evaluation, as demonstrated in `fn demo`
                // above.
                //
                // Okay if we have two-phase borrows. Note that even
                // if `self.cx.body(..)` holds onto a reference into
                // `self.cx`, `self.cx` is an immutable-borrow, so
                // nothing in the activation for `self.hash_expr(..)`
                // can interfere with that immutable borrow.
                self.hash_expr(&self.cx.body(eid).value);
            },
            _ => {}
        }
    }

    fn hash_expr_mut(&mut self, e: &Expr) {
        match *e {
            Expr::Closure(eid) => {
                // Not okay: the call to `self.cx_mut.body(eid)` might
                // hold on to some mutably borrowed state in
                // `self.cx_mut`, which would then interfere with the
                // eventual activation of the `self` mutable borrow
                // for `self.hash_expr(..)`
                self.hash_expr(&self.cx_mut.body(eid).value);
                //~^ ERROR cannot borrow `*self`
            },
            _ => {}
        }
    }
}

struct Session;
struct Config;
trait LateLintPass<'a> { }

struct TrivialPass;
impl TrivialPass {
    fn new(_: &Session) -> Self { TrivialPass }
    fn new_mut(_: &mut Session) -> Self { TrivialPass }
}

struct CapturePass<'a> { s: &'a Session }
impl<'a> CapturePass<'a> {
    fn new(s: &'a Session) -> Self { CapturePass { s } }
    fn new_mut(s: &'a mut Session) -> Self { CapturePass { s } }
}

impl<'a> LateLintPass<'a> for TrivialPass { }
impl<'a, 'b> LateLintPass<'a> for CapturePass<'b> { }

struct Registry<'a> { sess_mut: &'a mut Session }
impl<'a> Registry<'a> {
    fn register_static(&mut self, _: Box<dyn LateLintPass + 'static>) { }

    // Note: there isn't an interesting distinction between these
    // different methods explored by any of the cases in the test
    // below. pnkfelix just happened to write these cases out while
    // exploring variations on `dyn for <'a> Trait<'a> + 'static`, and
    // then decided to keep these particular ones in.
    fn register_bound(&mut self, _: Box<dyn LateLintPass + 'a>) { }
    fn register_univ(&mut self, _: Box<dyn for <'b> LateLintPass<'b> + 'a>) { }
    fn register_ref(&mut self, _: &dyn LateLintPass) { }
}

fn register_plugins<'a>(mk_reg: impl Fn() -> &'a mut Registry<'a>) {
    // Not okay without two-phase borrows: The implicit `&mut reg` of
    // the receiver is evaluaated first, and that conflicts with the
    // `reg.sess_mut` access during argument evaluation.
    //
    // Okay if we have two-phase borrows: inner borrows do not survive
    // to the actual method invocation, because `TrivialPass::new`
    // cannot (according to its type) keep them alive.
    let reg = mk_reg();
    reg.register_static(Box::new(TrivialPass::new(&reg.sess_mut)));
    let reg = mk_reg();
    reg.register_bound(Box::new(TrivialPass::new(&reg.sess_mut)));
    let reg = mk_reg();
    reg.register_univ(Box::new(TrivialPass::new(&reg.sess_mut)));
    let reg = mk_reg();
    reg.register_ref(&TrivialPass::new(&reg.sess_mut));

    // These are not okay: the inner mutable borrows immediately
    // conflict with the outer borrow/reservation, even with support
    // for two-phase borrows.
    let reg = mk_reg();
    reg.register_static(Box::new(TrivialPass::new(&mut reg.sess_mut)));
    //~^ ERROR cannot borrow `reg.sess_mut`
    let reg = mk_reg();
    reg.register_bound(Box::new(TrivialPass::new_mut(&mut reg.sess_mut)));
    //~^ ERROR cannot borrow `reg.sess_mut`
    let reg = mk_reg();
    reg.register_univ(Box::new(TrivialPass::new_mut(&mut reg.sess_mut)));
    //~^ ERROR cannot borrow `reg.sess_mut`
    let reg = mk_reg();
    reg.register_ref(&TrivialPass::new_mut(&mut reg.sess_mut));
    //~^ ERROR cannot borrow `reg.sess_mut`

    // These are not okay: the inner borrows may reach the actual
    // method invocation, because `CapturePass::new` might (according
    // to its type) keep them alive.
    //
    // (Also, we don't test `register_static` on CapturePass because
    // that will fail to get past lifetime inference.)
    let reg = mk_reg();
    reg.register_bound(Box::new(CapturePass::new(&reg.sess_mut)));
    //~^ ERROR cannot borrow `*reg` as mutable
    let reg = mk_reg();
    reg.register_univ(Box::new(CapturePass::new(&reg.sess_mut)));
    //~^ ERROR cannot borrow `*reg` as mutable
    let reg = mk_reg();
    reg.register_ref(&CapturePass::new(&reg.sess_mut));
    //~^ ERROR cannot borrow `*reg` as mutable

    // These are not okay: the inner mutable borrows immediately
    // conflict with the outer borrow/reservation, even with support
    // for two-phase borrows.
    //
    // (Again, we don't test `register_static` on CapturePass because
    // that will fail to get past lifetime inference.)
    let reg = mk_reg();
    reg.register_bound(Box::new(CapturePass::new_mut(&mut reg.sess_mut)));
    //~^ ERROR cannot borrow `reg.sess_mut` as mutable more than once at a time
    //~^^ ERROR cannot borrow `*reg` as mutable more than once at a time
    let reg = mk_reg();
    reg.register_univ(Box::new(CapturePass::new_mut(&mut reg.sess_mut)));
    //~^ ERROR cannot borrow `reg.sess_mut` as mutable more than once at a time
    //~^^ ERROR cannot borrow `*reg` as mutable more than once at a time
    let reg = mk_reg();
    reg.register_ref(&CapturePass::new_mut(&mut reg.sess_mut));
    //~^ ERROR cannot borrow `reg.sess_mut` as mutable more than once at a time
    //~^^ ERROR cannot borrow `*reg` as mutable more than once at a time
}

fn main() { }
