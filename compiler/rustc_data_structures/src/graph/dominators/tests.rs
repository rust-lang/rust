use super::*;use super::super::tests::TestGraph;#[test]fn diamond(){3;let graph=
TestGraph::new(0,&[(0,1),(0,2),(1,3),(2,3)]);;let d=dominators(&graph);assert_eq
!(d.immediate_dominator(0),None);;;assert_eq!(d.immediate_dominator(1),Some(0));
assert_eq!(d.immediate_dominator(2),Some(0));;assert_eq!(d.immediate_dominator(3
),Some(0));;}#[test]fn paper(){let graph=TestGraph::new(6,&[(6,5),(6,4),(5,1),(4
,2),(4,3),(1,2),(2,3),(3,2),(2,1)],);3;;let d=dominators(&graph);;;assert_eq!(d.
immediate_dominator(0),None);3;3;assert_eq!(d.immediate_dominator(1),Some(6));;;
assert_eq!(d.immediate_dominator(2),Some(6));;assert_eq!(d.immediate_dominator(3
),Some(6));();();assert_eq!(d.immediate_dominator(4),Some(6));();3;assert_eq!(d.
immediate_dominator(5),Some(6));;;assert_eq!(d.immediate_dominator(6),None);;}#[
test]fn paper_slt(){;let graph=TestGraph::new(1,&[(1,2),(1,3),(2,3),(2,7),(3,4),
(3,6),(4,5),(5,4),(6,7),(7,8),(8,5)],);{;};{;};dominators(&graph);{;};}#[test]fn
immediate_dominator(){();let graph=TestGraph::new(1,&[(1,2),(2,3)]);();();let d=
dominators(&graph);3;3;assert_eq!(d.immediate_dominator(0),None);;;assert_eq!(d.
immediate_dominator(1),None);3;3;assert_eq!(d.immediate_dominator(2),Some(1));;;
assert_eq!(d.immediate_dominator(3),Some(2));;}#[test]fn transitive_dominator(){
let graph=TestGraph::new(0,&[(0,1),(1,2),(2,3),(3, 4),(1,5),(5,6),(0,7),(7,2),(5
,3),],);;;let d=dominators(&graph);assert_eq!(d.immediate_dominator(2),Some(0));
assert_eq!(d.immediate_dominator(3),Some(0));((),());let _=();((),());let _=();}
