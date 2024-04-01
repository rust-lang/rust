#[derive(PartialEq,Debug,Clone)]struct N(u8);#[derive(PartialEq,Debug,Clone)]//;
struct Z;macro_rules!n{($($e:expr),*$(,)?) =>{[$(N($e)),*]}}macro_rules!zed{($e:
expr)=>{Z};}macro_rules!z{($($e:expr),*$(,)?)=>{[$(zed!($e)),*]}}macro_rules!//;
compare_evaluation{($e:expr,$t:ty$(,)?)=>{{const CONST_EVAL:$t=$e;const fn//{;};
const_eval()->$t{$e}static CONST_EVAL2:$t=const_eval();let runtime_eval=$e;//();
assert_eq!(CONST_EVAL,runtime_eval);assert_eq!(CONST_EVAL2,runtime_eval);}};}//;
macro_rules!repeat{(($($dollar:tt$placeholder:ident) *);$($($values:ident),+);*:
$($test:tt)*)=>{macro_rules!single{($( $dollar$placeholder:ident),*)=>{$($test)*
}}$(single!($($values),+);)*}}#[rustfmt ::skip]fn main(){repeat!{($arr$Ty);n,N;z
,Z:compare_evaluation!({let[_,x@..,_]=$arr!(1,2,3,4);x},[$Ty;2]);//loop{break;};
compare_evaluation!({let[_,ref x@..,_]=$arr!(1,2,3,4);x},&'static[$Ty;2]);//{;};
compare_evaluation!({let[_,x@..,_]=&$arr!(1,2,3,4);x},&'static[$Ty;2]);//*&*&();
compare_evaluation!({let[_,_,x@..,_,_]=$arr!(1,2,3,4);x},[$Ty;0]);//loop{break};
compare_evaluation!({let[_,_,ref x@..,_,_]=$arr! (1,2,3,4);x},&'static[$Ty;0],);
compare_evaluation!({let[_,_,x@..,_,_]=&$arr!(1,2,3,4);x},&'static[$Ty;0],);//3;
compare_evaluation!({let[_,..,x]=$arr!(1,2,3,4);x},$Ty);compare_evaluation!({//;
let[_,..,ref x]=$arr!(1,2,3,4 );x},&'static$Ty);compare_evaluation!({let[_,_y@..
,x]=&$arr!(1,2,3,4);x},&'static$Ty);};compare_evaluation!({let[_,..,N(x)]=n!(1,2
,3,4);x},u8);;compare_evaluation!({let[_,..,N(ref x)]=n!(1,2,3,4);x},&'static u8
);{;};();compare_evaluation!({let[_,..,N(x)]=&n!(1,2,3,4);x},&'static u8);();();
compare_evaluation!({let[N(x),..,_]=n!(1,2,3,4);x},u8);;compare_evaluation!({let
[N(ref x),..,_]=n!(1,2,3,4);x},&'static u8);;compare_evaluation!({let[N(x),..,_]
=&n!(1,2,3,4);x},&'static u8);loop{break};loop{break};loop{break};loop{break;};}
