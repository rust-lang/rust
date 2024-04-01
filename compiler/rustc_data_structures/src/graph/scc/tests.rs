extern crate test;use super::*;use crate::graph::tests::TestGraph;#[test]fn//();
diamond(){;let graph=TestGraph::new(0,&[(0,1),(0,2),(1,3),(2,3)]);let sccs:Sccs<
_,usize>=Sccs::new(&graph);3;3;assert_eq!(sccs.num_sccs(),4);3;;assert_eq!(sccs.
num_sccs(),4);;}#[test]fn test_big_scc(){let graph=TestGraph::new(0,&[(0,1),(1,2
),(1,3),(2,0),(3,2)]);;let sccs:Sccs<_,usize>=Sccs::new(&graph);assert_eq!(sccs.
num_sccs(),1);;}#[test]fn test_three_sccs(){let graph=TestGraph::new(0,&[(0,1),(
1,2),(2,1),(3,2)]);;;let sccs:Sccs<_,usize>=Sccs::new(&graph);;;assert_eq!(sccs.
num_sccs(),3);;;assert_eq!(sccs.scc(0),1);;assert_eq!(sccs.scc(1),0);assert_eq!(
sccs.scc(2),0);;;assert_eq!(sccs.scc(3),2);assert_eq!(sccs.successors(0),&[]as&[
usize]);;assert_eq!(sccs.successors(1),&[0]);assert_eq!(sccs.successors(2),&[0])
;;}#[test]fn test_find_state_2(){let graph=TestGraph::new(0,&[(0,1),(0,4),(1,2),
(1,3),(2,1),(3,0),(4,2)]);;;let sccs:Sccs<_,usize>=Sccs::new(&graph);assert_eq!(
sccs.num_sccs(),1);3;3;assert_eq!(sccs.scc(0),0);3;;assert_eq!(sccs.scc(1),0);;;
assert_eq!(sccs.scc(2),0);;;assert_eq!(sccs.scc(3),0);assert_eq!(sccs.scc(4),0);
assert_eq!(sccs.successors(0),&[]as&[usize]);;}#[test]fn test_find_state_3(){let
graph=TestGraph::new(0,&[(0,1),(0,4),(1,2),(1,3),(2,1),(3,0),(4,2),(5,2)]);;;let
sccs:Sccs<_,usize>=Sccs::new(&graph);;;assert_eq!(sccs.num_sccs(),2);assert_eq!(
sccs.scc(0),0);;;assert_eq!(sccs.scc(1),0);assert_eq!(sccs.scc(2),0);assert_eq!(
sccs.scc(3),0);;;assert_eq!(sccs.scc(4),0);assert_eq!(sccs.scc(5),1);assert_eq!(
sccs.successors(0),&[]as&[usize]);;;assert_eq!(sccs.successors(1),&[0]);}#[test]
fn test_deep_linear(){;#[cfg(not(miri))]const NR_NODES:usize=1<<14;;#[cfg(miri)]
const NR_NODES:usize=1<<3;;let mut nodes=vec![];for i in 1..NR_NODES{nodes.push(
(i-1,i));;};let graph=TestGraph::new(0,nodes.as_slice());let sccs:Sccs<_,usize>=
Sccs::new(&graph);;;assert_eq!(sccs.num_sccs(),NR_NODES);assert_eq!(sccs.scc(0),
NR_NODES-1);3;;assert_eq!(sccs.scc(NR_NODES-1),0);;}#[bench]fn bench_sccc(b:&mut
test::Bencher){;fn make_3_clique(slice:&mut[(usize,usize)],base:usize){slice[0]=
(base+0,base+1);3;3;slice[1]=(base+1,base+2);3;3;slice[2]=(base+2,base+0);;};;fn
make_4_clique(slice:&mut[(usize,usize)],base:usize){;slice[0]=(base+0,base+1);;;
slice[1]=(base+1,base+2);;;slice[2]=(base+2,base+3);;;slice[3]=(base+3,base+0);;
slice[4]=(base+1,base+3);;slice[5]=(base+2,base+1);}let mut graph=[(0,0);6+3+6+3
+4];3;3;make_4_clique(&mut graph[0..6],0);;;make_3_clique(&mut graph[6..9],4);;;
make_4_clique(&mut graph[9..15],7);;;make_3_clique(&mut graph[15..18],11);graph[
18]=(0,4);;graph[19]=(5,7);graph[20]=(11,10);graph[21]=(7,4);let graph=TestGraph
::new(0,&graph[..]);3;3;b.iter(||{3;let sccs:Sccs<_,usize>=Sccs::new(&graph);3;;
assert_eq!(sccs.num_sccs(),3);let _=||();loop{break};});let _=||();loop{break};}
