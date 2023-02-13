use test::black_box;
use test::Bencher;

macro_rules! map_array {
    ($func_name:ident, $start_item: expr, $map_item: expr, $arr_size: expr) => {
        #[bench]
        fn $func_name(b: &mut Bencher) {
            let arr = [$start_item; $arr_size];
            b.iter(|| black_box(arr).map(|_| black_box($map_item)));
        }
    };
}

map_array!(map_8byte_8byte_8, 0u64, 1u64, 800);
map_array!(map_8byte_8byte_64, 0u64, 1u64, 6400);
map_array!(map_8byte_8byte_256, 0u64, 1u64, 25600);

map_array!(map_8byte_256byte_256, 0u64, [0u64; 4], 25600);
map_array!(map_256byte_8byte_256, [0u64; 4], 0u64, 25600);
