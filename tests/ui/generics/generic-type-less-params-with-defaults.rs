use std::marker;

struct Heap;

struct Vec<T, A = Heap>(
    marker::PhantomData<(T,A)>);

struct HashMap<K, V, S = ()>(marker::PhantomData<(K,V,S)>);

fn main() {
    let _: Vec;
    //~^ ERROR missing generics for struct `Vec`
    //~| SUGGESTION <T>

    let _x = (1..10).collect::<HashMap>();
    //~^ ERROR missing generics for struct `HashMap`
    //~| SUGGESTION <_, _>

    ().extend::<[(); 0]>({
        fn not_the_extend() {
            let _: Vec;
            //~^ ERROR missing generics for struct `Vec`
            //~| SUGGESTION <T>
        }
        []
    });
}
