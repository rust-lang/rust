//@ normalize-stderr: "DefId\(.+?\)" -> "DefId(..)"

#![feature(rustc_attrs)]
#![feature(trait_alias)]
#![feature(stmt_expr_attributes)]

#[rustc_dump_generics]
trait NiceOfTheFoundation<'_a: '_a, '_b, const N: usize, T, U: Clone> {
    //~^ ERROR: rustc_dump_generics: DefId(0:3 ~ dump_generics[c1f6]::NiceOfTheFoundation)

    #[rustc_dump_generics]
    type ToInviteUs<V: PartialEq>;
    //~^ ERROR: rustc_dump_generics: DefId(0:9 ~ dump_generics[c1f6]::NiceOfTheFoundation::ToInviteUs)

    #[rustc_dump_generics]
    const OVER_FOR: usize;
    //~^ ERROR: rustc_dump_generics: DefId(0:11 ~ dump_generics[c1f6]::NiceOfTheFoundation::OVER_FOR)

    #[rustc_dump_generics]
    fn a_picnic<const NN: usize>( //~ ERROR: rustc_dump_generics: DefId(0:12 ~ dump_generics[c1f6]::NiceOfTheFoundation::a_picnic)
        &self,
        eh: [u8; N],
        ferris: [u32; NN]
    ) { }
}

#[rustc_dump_generics]
pub trait IHopeItMadeLotsOfCode<'a, 'b, const N: usize, T, U: Clone>
//~^ ERROR: rustc_dump_generics: DefId(0:16 ~ dump_generics[c1f6]::IHopeItMadeLotsOfCode)
    = NiceOfTheFoundation<'a, 'b, N, T, U>;

#[rustc_dump_generics]
struct LookItsFromCorro<'a, 'b, const N: usize, T, U: Clone> {
    //~^ ERROR: rustc_dump_generics: DefId(0:22 ~ dump_generics[c1f6]::LookItsFromCorro)
    dear_pesky_engineers: &'a T,
    the_rustaceans_and_i: &'b [U; N]
}

#[rustc_dump_generics]
enum HaveTakenOver<'a: 'a, 'b, const N: usize, T, U: Clone> {
    //~^ ERROR: rustc_dump_generics: DefId(0:31 ~ dump_generics[c1f6]::HaveTakenOver)
    The(&'a T),
    CratesIo(&'b U),
    Kingdom([(); N]),
}

#[rustc_dump_generics]
union TheFoundation<'a: 'a, 'b, const N: usize, T, U: Clone> {
    //~^ ERROR: rustc_dump_generics: DefId(0:47 ~ dump_generics[c1f6]::TheFoundation)
    is_now_a_permanent_guest: &'a T,
    at_one_of_my_seven_pull_requests: &'b [U; N],
}

#[rustc_dump_generics]
const fn i_dare_you<'a: 'a, 'b, const N: usize, T, U>(you_can: &'a bool, _: &'b [(T, U); N]) {
    //~^ ERROR: rustc_dump_generics: DefId(0:56 ~ dump_generics[c1f6]::i_dare_you)
    let _to_find_it = if *you_can { 1 } else { 2 };

    let we_got_to_find_the_foundation =
        #[rustc_dump_generics]
        || {};
        // and you gotta help us!
}

#[rustc_dump_generics]
trait IfYouNeed<'_a: '_a, '_b, const N: usize, T, U: Clone> {}
//~^ ERROR: rustc_dump_generics: DefId(0:64 ~ dump_generics[c1f6]::IfYouNeed)

#[rustc_dump_generics]
type Instructions<'a: 'a, 'b, const N: usize, T, U: Clone> = dyn IfYouNeed<'a, 'b, N, T, U>;
//~^ ERROR: rustc_dump_generics: DefId(0:70 ~ dump_generics[c1f6]::Instructions)

#[rustc_dump_generics]
const ON_HOW_TO_GET: usize = <() as NiceOfTheFoundation::<'static, 'static, 7, (), ()>>::OVER_FOR;
//~^ ERROR: rustc_dump_generics: DefId(0:76 ~ dump_generics[c1f6]::ON_HOW_TO_GET)


// FIXME: make sure we have tests for these targets.
// Allow(Target::Impl { of_trait: false }),
// Allow(Target::Impl { of_trait: true }),
// Allow(Target::Method(MethodKind::Inherent)),
// Allow(Target::Method(MethodKind::Trait { body: false })),
// Allow(Target::Method(MethodKind::Trait { body: true })),
// Allow(Target::Method(MethodKind::TraitImpl)),
// Allow(Target::Delegation { mac: false }),
// Allow(Target::Delegation { mac: true }),

impl<'_a: '_a, '_b, const N: usize, T, U: Clone> NiceOfTheFoundation<'_a, '_b, N, T, U> for () {
    type ToInviteUs<V: PartialEq> = usize;

    const OVER_FOR: usize = 7;
}

fn main() {}
