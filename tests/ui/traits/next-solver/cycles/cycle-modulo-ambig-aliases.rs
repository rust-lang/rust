//@ compile-flags: -Znext-solver

// A regression test for #125269. We previously ended up
// recursively proving `&<_ as SpeciesPackedElem>::Assoc: Typed`
// for all aliases which ended up causing exponential blowup.
//
// This has been fixed by eagerly normalizing the associated
// type before computing the nested goals, resulting in an
// immediate inductive cycle.

pub trait Typed {}

pub struct SpeciesCases<E>(E);

pub trait SpeciesPackedElim {
    type Ogre;
    type Cyclops;
    type Wendigo;
    type Cavetroll;
    type Mountaintroll;
    type Swamptroll;
    type Dullahan;
    type Werewolf;
    type Occultsaurok;
    type Mightysaurok;
    type Slysaurok;
    type Mindflayer;
    type Minotaur;
    type Tidalwarrior;
    type Yeti;
    type Harvester;
    type Blueoni;
    type Redoni;
    type Cultistwarlord;
    type Cultistwarlock;
    type Huskbrute;
    type Tursus;
    type Gigasfrost;
    type AdletElder;
    type SeaBishop;
    type HaniwaGeneral;
    type TerracottaBesieger;
    type TerracottaDemolisher;
    type TerracottaPunisher;
    type TerracottaPursuer;
    type Cursekeeper;
}

impl<'b, E: SpeciesPackedElim> Typed for &'b SpeciesCases<E>
where
    &'b E::Ogre: Typed,
    &'b E::Cyclops: Typed,
    &'b E::Wendigo: Typed,
    &'b E::Cavetroll: Typed,
    &'b E::Mountaintroll: Typed,
    &'b E::Swamptroll: Typed,
    &'b E::Dullahan: Typed,
    &'b E::Werewolf: Typed,
    &'b E::Occultsaurok: Typed,
    &'b E::Mightysaurok: Typed,
    &'b E::Slysaurok: Typed,
    &'b E::Mindflayer: Typed,
    &'b E::Minotaur: Typed,
    &'b E::Tidalwarrior: Typed,
    &'b E::Yeti: Typed,
    &'b E::Harvester: Typed,
    &'b E::Blueoni: Typed,
    &'b E::Redoni: Typed,
    &'b E::Cultistwarlord: Typed,
    &'b E::Cultistwarlock: Typed,
    &'b E::Huskbrute: Typed,
    &'b E::Tursus: Typed,
    &'b E::Gigasfrost: Typed,
    &'b E::AdletElder: Typed,
    &'b E::SeaBishop: Typed,
    &'b E::HaniwaGeneral: Typed,
    &'b E::TerracottaBesieger: Typed,
    &'b E::TerracottaDemolisher: Typed,
    &'b E::TerracottaPunisher: Typed,
    &'b E::TerracottaPursuer: Typed,
    &'b E::Cursekeeper: Typed,
{}

fn foo<T: Typed>() {}

fn main() {
    foo::<&_>();
    //~^ ERROR overflow evaluating the requirement `&_: Typed`
}
