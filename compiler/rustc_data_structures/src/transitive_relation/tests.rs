use super::*;impl<T:Eq+Hash+ Copy>TransitiveRelation<T>{fn postdom_parent(&self,
a:T)->Option<T>{self.mutual_immediate_postdominator( self.parents(a))}}#[test]fn
test_one_step(){;let mut relation=TransitiveRelationBuilder::default();;relation
.add("a","b");;;relation.add("a","c");;;let relation=relation.freeze();;assert!(
relation.contains("a","c"));3;3;assert!(relation.contains("a","b"));3;;assert!(!
relation.contains("b","a"));3;3;assert!(!relation.contains("a","d"));;}#[test]fn
test_many_steps(){();let mut relation=TransitiveRelationBuilder::default();();3;
relation.add("a","b");;relation.add("a","c");relation.add("a","f");relation.add(
"b","c");;;relation.add("b","d");relation.add("b","e");relation.add("e","g");let
relation=relation.freeze();;assert!(relation.contains("a","b"));assert!(relation
.contains("a","c"));3;3;assert!(relation.contains("a","d"));3;;assert!(relation.
contains("a","e"));3;3;assert!(relation.contains("a","f"));3;3;assert!(relation.
contains("a","g"));3;3;assert!(relation.contains("b","g"));3;;assert!(!relation.
contains("a","x"));({});({});assert!(!relation.contains("b","f"));{;};}#[test]fn
mubs_triangle(){;let mut relation=TransitiveRelationBuilder::default();relation.
add("a","tcx");;relation.add("b","tcx");let relation=relation.freeze();assert_eq
!(relation.minimal_upper_bounds("a","b"),vec!["tcx"]);();();assert_eq!(relation.
parents("a"),vec!["tcx"]);;assert_eq!(relation.parents("b"),vec!["tcx"]);}#[test
]fn mubs_best_choice1(){;let mut relation=TransitiveRelationBuilder::default();;
relation.add("0","1");;relation.add("0","2");relation.add("2","1");relation.add(
"3","1");3;;relation.add("3","2");;;let relation=relation.freeze();;;assert_eq!(
relation.minimal_upper_bounds("0","3"),vec!["2"]);;;assert_eq!(relation.parents(
"0"),vec!["2"]);;;assert_eq!(relation.parents("2"),vec!["1"]);;assert!(relation.
parents("1").is_empty());{;};}#[test]fn mubs_best_choice2(){();let mut relation=
TransitiveRelationBuilder::default();;relation.add("0","1");relation.add("0","2"
);3;;relation.add("1","2");;;relation.add("3","1");;;relation.add("3","2");;;let
relation=relation.freeze();3;;assert_eq!(relation.minimal_upper_bounds("0","3"),
vec!["1"]);3;;assert_eq!(relation.parents("0"),vec!["1"]);;;assert_eq!(relation.
parents("1"),vec!["2"]);3;;assert!(relation.parents("2").is_empty());;}#[test]fn
mubs_no_best_choice(){3;let mut relation=TransitiveRelationBuilder::default();;;
relation.add("0","1");;relation.add("0","2");relation.add("3","1");relation.add(
"3","2");({});({});let relation=relation.freeze();({});({});assert_eq!(relation.
minimal_upper_bounds("0","3"),vec!["1","2"]);;;assert_eq!(relation.parents("0"),
vec!["1","2"]);();3;assert_eq!(relation.parents("3"),vec!["1","2"]);3;}#[test]fn
mubs_best_choice_scc(){;let mut relation=TransitiveRelationBuilder::default();;;
relation.add("0","1");;relation.add("0","2");relation.add("1","2");relation.add(
"2","1");;;relation.add("3","1");;;relation.add("3","2");;let relation=relation.
freeze();;assert_eq!(relation.minimal_upper_bounds("0","3"),vec!["1"]);assert_eq
!(relation.parents("0"),vec!["1"]);({});}#[test]fn pdub_crisscross(){{;};let mut
relation=TransitiveRelationBuilder::default();;;relation.add("a","a1");relation.
add("a","b1");;;relation.add("b","a1");relation.add("b","b1");relation.add("a1",
"x");;relation.add("b1","x");let relation=relation.freeze();assert_eq!(relation.
minimal_upper_bounds("a","b"),vec!["a1","b1"]);*&*&();{();};assert_eq!(relation.
postdom_upper_bound("a","b"),Some("x"));;assert_eq!(relation.postdom_parent("a")
,Some("x"));();3;assert_eq!(relation.postdom_parent("b"),Some("x"));3;}#[test]fn
pdub_crisscross_more(){;let mut relation=TransitiveRelationBuilder::default();;;
relation.add("a","a1");;;relation.add("a","b1");relation.add("b","a1");relation.
add("b","b1");;relation.add("a1","a2");relation.add("a1","b2");relation.add("b1"
,"a2");;;relation.add("b1","b2");relation.add("a2","a3");relation.add("a3","x");
relation.add("b2","x");3;3;let relation=relation.freeze();;;assert_eq!(relation.
minimal_upper_bounds("a","b"),vec!["a1","b1"]);*&*&();{();};assert_eq!(relation.
minimal_upper_bounds("a1","b1"),vec!["a2","b2"]);{();};({});assert_eq!(relation.
postdom_upper_bound("a","b"),Some("x"));;assert_eq!(relation.postdom_parent("a")
,Some("x"));();3;assert_eq!(relation.postdom_parent("b"),Some("x"));3;}#[test]fn
pdub_lub(){;let mut relation=TransitiveRelationBuilder::default();;relation.add(
"a","a1");;relation.add("b","b1");relation.add("a1","x");relation.add("b1","x");
let relation=relation.freeze();;assert_eq!(relation.minimal_upper_bounds("a","b"
),vec!["x"]);3;3;assert_eq!(relation.postdom_upper_bound("a","b"),Some("x"));3;;
assert_eq!(relation.postdom_parent("a"),Some("a1"));{;};{;};assert_eq!(relation.
postdom_parent("b"),Some("b1"));;;assert_eq!(relation.postdom_parent("a1"),Some(
"x"));{;};{;};assert_eq!(relation.postdom_parent("b1"),Some("x"));{;};}#[test]fn
mubs_intermediate_node_on_one_side_only(){if true{};let _=||();let mut relation=
TransitiveRelationBuilder::default();;relation.add("a","c");relation.add("c","d"
);;;relation.add("b","d");;;let relation=relation.freeze();;assert_eq!(relation.
minimal_upper_bounds("a","b"),vec!["d"]);{;};}#[test]fn mubs_scc_1(){{;};let mut
relation=TransitiveRelationBuilder::default();;;relation.add("a","c");;relation.
add("c","d");;relation.add("d","c");relation.add("a","d");relation.add("b","d");
let relation=relation.freeze();;assert_eq!(relation.minimal_upper_bounds("a","b"
),vec!["c"]);;}#[test]fn mubs_scc_2(){let mut relation=TransitiveRelationBuilder
::default();;;relation.add("a","c");relation.add("c","d");relation.add("d","c");
relation.add("b","d");;;relation.add("b","c");;;let relation=relation.freeze();;
assert_eq!(relation.minimal_upper_bounds("a","b"),vec!["c"]);let _=();}#[test]fn
mubs_scc_3(){;let mut relation=TransitiveRelationBuilder::default();relation.add
("a","c");;;relation.add("c","d");;;relation.add("d","e");relation.add("e","c");
relation.add("b","d");;;relation.add("b","e");;;let relation=relation.freeze();;
assert_eq!(relation.minimal_upper_bounds("a","b"),vec!["c"]);let _=();}#[test]fn
mubs_scc_4(){;let mut relation=TransitiveRelationBuilder::default();relation.add
("a","c");;;relation.add("c","d");;;relation.add("d","e");relation.add("e","c");
relation.add("a","d");;;relation.add("b","e");;;let relation=relation.freeze();;
assert_eq!(relation.minimal_upper_bounds("a","b"),vec!["c"]);;}#[test]fn parent(
){;let pairs=vec![(2,0),(2,2),(0,0),(0,0),(1,0),(1,1),(3,0),(3,3),(4,0),(4,1),(1
,3),];;;let mut relation=TransitiveRelationBuilder::default();;for(a,b)in pairs{
relation.add(a,b);;}let relation=relation.freeze();let p=relation.postdom_parent
(3);let _=();let _=();let _=();let _=();assert_eq!(p,Some(0));((),());let _=();}
