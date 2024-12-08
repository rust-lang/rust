// ignore-tidy-linelength

//@ set loud_id = "$.index[*][?(@.name=='Loud')].id"
pub trait Loud {}

//@ set very_loud_id = "$.index[*][?(@.name=='VeryLoud')].id"
//@ count "$.index[*][?(@.name=='VeryLoud')].inner.trait.bounds[*]" 1
//@ is    "$.index[*][?(@.name=='VeryLoud')].inner.trait.bounds[0].trait_bound.trait.id" $loud_id
pub trait VeryLoud: Loud {}

//@ set sounds_good_id = "$.index[*][?(@.name=='SoundsGood')].id"
pub trait SoundsGood {}

//@ count "$.index[*][?(@.name=='MetalBand')].inner.trait.bounds[*]" 2
//@ is    "$.index[*][?(@.name=='MetalBand')].inner.trait.bounds[0].trait_bound.trait.id" $very_loud_id
//@ is    "$.index[*][?(@.name=='MetalBand')].inner.trait.bounds[1].trait_bound.trait.id" $sounds_good_id
pub trait MetalBand: VeryLoud + SoundsGood {}

//@ count "$.index[*][?(@.name=='DnabLatem')].inner.trait.bounds[*]" 2
//@ is    "$.index[*][?(@.name=='DnabLatem')].inner.trait.bounds[1].trait_bound.trait.id" $very_loud_id
//@ is    "$.index[*][?(@.name=='DnabLatem')].inner.trait.bounds[0].trait_bound.trait.id" $sounds_good_id
pub trait DnabLatem: SoundsGood + VeryLoud {}
