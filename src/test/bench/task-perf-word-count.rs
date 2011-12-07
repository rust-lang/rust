/**
   A parallel word-frequency counting program.

   This is meant primarily to demonstrate Rust's MapReduce framework.

   It takes a list of files on the command line and outputs a list of
   words along with how many times each word is used.

*/

use std;

import option = std::option::t;
import std::option::some;
import std::option::none;
import std::str;
import std::map;
import std::vec;
import std::io;

import std::time;
import std::u64;
import std::result;

import std::task;
import std::task::joinable_task;
import std::comm;
import std::comm::chan;
import std::comm::port;
import std::comm::recv;
import std::comm::send;

fn map(input: str, emit: map_reduce::putter) {
    let f = io::string_reader(input);


    while true {
        alt read_word(f) { some(w) { emit(w, 1); } none. { break; } }
    }
}

fn reduce(_word: str, get: map_reduce::getter) {
    let count = 0;

    while true { alt get() { some(_) { count += 1; } none. { break; } } }
}

mod map_reduce {
    export putter;
    export getter;
    export mapper;
    export reducer;
    export map_reduce;

    type putter = fn@(str, int);

    type mapper = fn(str, putter);

    type getter = fn@() -> option<int>;

    type reducer = fn(str, getter);

    tag ctrl_proto {
        find_reducer(str, chan<chan<reduce_proto>>);
        mapper_done;
    }

    tag reduce_proto { emit_val(int); done; ref; release; }

    fn start_mappers(ctrl: chan<ctrl_proto>, -inputs: [str]) ->
       [joinable_task] {
        let tasks = [];
        for i: str in inputs {
            tasks += [task::spawn_joinable((ctrl, i), map_task)];
        }
        ret tasks;
    }

    fn map_task(args: (chan<ctrl_proto>, str)) {
        let (ctrl, input) = args;
        // log_err "map_task " + input;
        let intermediates = map::new_str_hash();

        fn emit(im: map::hashmap<str, chan<reduce_proto>>,
                ctrl: chan<ctrl_proto>, key: str, val: int) {
            let c;
            alt im.find(key) {
              some(_c) {
                c = _c;
              }
              none. {
                let p = port();
                send(ctrl, find_reducer(key, chan(p)));
                c = recv(p);
                im.insert(key, c);
                send(c, ref);
              }
            }
            send(c, emit_val(val));
        }

        map(input, bind emit(intermediates, ctrl, _, _));

        intermediates.values {|v| send(v, release); }

        send(ctrl, mapper_done);
    }

    fn reduce_task(args: (str, chan<chan<reduce_proto>>)) {
        let (key, out) = args;
        let p = port();

        send(out, chan(p));

        let state = @{mutable ref_count: 0, mutable is_done: false};

        fn get(p: port<reduce_proto>, state: @{mutable ref_count: int,
                                               mutable is_done: bool})
            -> option<int> {
            while !state.is_done || state.ref_count > 0 {
                alt recv(p) {
                  emit_val(v) {
                    // log_err #fmt("received %d", v);
                    ret some(v);
                  }
                  done. {
                    // log_err "all done";
                    state.is_done = true;
                  }
                  ref. { state.ref_count += 1; }
                  release. { state.ref_count -= 1; }
                }
            }
            ret none;
        }

        reduce(key, bind get(p, state));
    }

    fn map_reduce(-inputs: [str]) {
        let ctrl = port::<ctrl_proto>();

        // This task becomes the master control task. It task::_spawns
        // to do the rest.

        let reducers: map::hashmap<str, chan<reduce_proto>>;

        reducers = map::new_str_hash();

        let num_mappers = vec::len(inputs) as int;
        let tasks = start_mappers(chan(ctrl), inputs);

        while num_mappers > 0 {
            alt recv(ctrl) {
              mapper_done. {
                // log_err "received mapper terminated.";
                num_mappers -= 1;
              }
              find_reducer(k, cc) {
                let c;
                // log_err "finding reducer for " + k;
                alt reducers.find(k) {
                  some(_c) {
                    // log_err "reusing existing reducer for " + k;
                    c = _c;
                  }
                  none. {
                    // log_err "creating new reducer for " + k;
                    let p = port();
                    tasks +=
                        [task::spawn_joinable((k, chan(p)), reduce_task)];
                    c = recv(p);
                    reducers.insert(k, c);
                  }
                }
                send(cc, c);
              }
            }
        }

        reducers.values {|v| send(v, done); }

        for t in tasks { task::join(t); }
    }
}

fn main(argv: [str]) {
    // We can get by with 8k stacks, and we'll probably exhaust our
    // address space otherwise.
    task::set_min_stack(8192u);

    let inputs = if vec::len(argv) < 2u {
        [input1(), input2(), input3()]
    } else {
        vec::map({|f| result::get(io::read_whole_file_str(f)) },
                 vec::slice(argv, 1u, vec::len(argv)))
    };

    let start = time::precise_time_ns();

    map_reduce::map_reduce(inputs);
    let stop = time::precise_time_ns();

    let elapsed = stop - start;
    elapsed /= 1000000u64;

    log_err "MapReduce completed in " + u64::str(elapsed) + "ms";
}

fn read_word(r: io::reader) -> option<str> {
    let w = "";

    while !r.eof() {
        let c = r.read_char();


        if is_word_char(c) {
            w += str::from_char(c);
        } else { if w != "" { ret some(w); } }
    }
    ret none;
}

fn is_digit(c: char) -> bool {
    alt c {
      '0' { true }
      '1' { true }
      '2' { true }
      '3' { true }
      '4' { true }
      '5' { true }
      '6' { true }
      '7' { true }
      '8' { true }
      '9' { true }
      _ { false }
    }
}

fn is_alpha_lower(c: char) -> bool {
    alt c {
      'a' { true }
      'b' { true }
      'c' { true }
      'd' { true }
      'e' { true }
      'f' { true }
      'g' { true }
      'h' { true }
      'i' { true }
      'j' { true }
      'k' { true }
      'l' { true }
      'm' { true }
      'n' { true }
      'o' { true }
      'p' { true }
      'q' { true }
      'r' { true }
      's' { true }
      't' { true }
      'u' { true }
      'v' { true }
      'w' { true }
      'x' { true }
      'y' { true }
      'z' { true }
      _ { false }
    }
}

fn is_alpha_upper(c: char) -> bool {
    alt c {
      'A' { true }
      'B' { true }
      'C' { true }
      'D' { true }
      'E' { true }
      'F' { true }
      'G' { true }
      'H' { true }
      'I' { true }
      'J' { true }
      'K' { true }
      'L' { true }
      'M' { true }
      'N' { true }
      'O' { true }
      'P' { true }
      'Q' { true }
      'R' { true }
      'S' { true }
      'T' { true }
      'U' { true }
      'V' { true }
      'W' { true }
      'X' { true }
      'Y' { true }
      'Z' { true }
      _ { false }
    }
}

fn is_alpha(c: char) -> bool { is_alpha_upper(c) || is_alpha_lower(c) }

fn is_word_char(c: char) -> bool { is_alpha(c) || is_digit(c) || c == '_' }



fn input1() -> str { " Lorem ipsum dolor sit amet, consectetur
adipiscing elit. Vestibulum tempor erat a dui commodo congue. Proin ac
imperdiet est. Nunc volutpat placerat justo, ac euismod nisl elementum
et. Nam a eros eleifend dolor porttitor auctor a a felis. Maecenas dui
odio, malesuada eget bibendum at, ultrices suscipit enim. Sed libero
dolor, sagittis eget mattis quis, imperdiet quis diam. Praesent eu
tristique nunc. Integer blandit commodo elementum. In eros lacus,
pretium vel fermentum vitae, euismod ut nulla.

Cras eget magna tempor mauris gravida laoreet. Suspendisse venenatis
volutpat molestie. Pellentesque suscipit nisl feugiat sem blandit
venenatis. Mauris id odio nec est elementum congue sed id
diam. Maecenas viverra, mi id aliquam commodo, ipsum dolor iaculis
odio, sed fringilla neque ipsum quis orci. Pellentesque dui dolor,
faucibus a rutrum sed, faucibus a mi. In eget sodales
ipsum. Pellentesque sollicitudin dapibus diam, ac interdum tellus
porta ac.

Donec ligula mi, sodales vel cursus a, dapibus ut sapien. In convallis
tempor libero, id dapibus mi sodales quis. Suspendisse
potenti. Vestibulum feugiat bibendum bibendum. Maecenas metus magna,
consequat in mollis at, malesuada id sem. Donec interdum viverra enim
nec ornare. Donec pellentesque neque magna.

Donec euismod, ante quis tempor pretium, leo lectus ornare arcu, sed
porttitor nisl ipsum elementum lectus. Nam rhoncus dictum sapien sed
tincidunt. Integer sit amet dui orci. Quisque lectus elit, dignissim
eget mattis nec, cursus nec erat. Fusce vitae metus nulla, et mattis
quam. Nullam sit amet diam augue. Nunc non ante eu enim lacinia
condimentum ac eget lectus.

Aliquam ut pulvinar tellus. Vestibulum ante ipsum primis in faucibus
orci luctus et ultrices posuere cubilia Curae; Pellentesque non urna
urna. Nulla facilisi. Aenean in felis quis massa aliquam eleifend non
sed libero. Proin sit amet iaculis urna. In hac habitasse platea
dictumst. Aenean scelerisque aliquet dolor, sit amet viverra est
laoreet nec. Curabitur non urna a augue rhoncus pulvinar. Integer
placerat vehicula nisl sed egestas. Morbi iaculis diam at erat
sollicitudin nec interdum libero tristique.  " }

fn input2() -> str { " Lorem ipsum dolor sit amet, consectetur
adipiscing elit. Proin enim nibh, scelerisque faucibus accumsan id,
feugiat id ipsum. In luctus mauris a massa consequat dignissim. Donec
sit amet sem urna. Nullam pellentesque accumsan mi, at convallis arcu
pharetra in. Quisque euismod gravida nibh in rutrum. Phasellus laoreet
elit porta augue molestie nec imperdiet quam venenatis. Maecenas et
egestas arcu. Donec vulputate mauris enim. Aenean malesuada urna sed
dui eleifend quis posuere massa malesuada. Proin varius fringilla
feugiat. Donec mollis lorem sit amet ligula blandit quis fermentum dui
eleifend. Fusce molestie sodales magna in mattis. Aenean imperdiet,
elit sit amet accumsan vehicula, velit massa semper nibh, et varius
justo sem ut orci. Sed et magna lectus. Vestibulum vehicula, tellus
non dapibus mattis, libero ligula ullamcorper odio, in interdum odio
sem at mi.

Donec ut rhoncus mi. Donec ullamcorper, sem nec laoreet ullamcorper,
metus metus accumsan orci, ac luctus est velit a dolor. Donec eros
lectus, facilisis ut volutpat sit amet, pellentesque eu
velit. Praesent eget nibh et arcu vestibulum consequat. Pellentesque
habitant morbi tristique senectus et netus et malesuada fames ac
turpis egestas. Pellentesque lectus est, rhoncus ut cursus sit amet,
hendrerit quis dui. Maecenas vel purus in tellus luctus semper vel non
orci. Proin viverra, erat eget pretium ultrices, quam quam vulputate
tortor, eu dapibus risus nunc ac ipsum. Vestibulum ante ipsum primis
in faucibus orci luctus et ultrices posuere cubilia Curae; Ut aliquet
augue volutpat arcu mattis ullamcorper. Quisque vulputate consectetur
massa, quis cursus mauris lacinia vitae. Morbi id mi eu leo accumsan
aliquet ac et arcu. Quisque risus nisi, rhoncus vulputate egestas sed,
rhoncus quis risus. Sed semper odio sed nulla accumsan vitae auctor
tortor mattis.

Vivamus vitae mauris turpis. Praesent consectetur mi non sem lacinia a
cursus sapien gravida. Aenean viverra turpis sit amet ligula
vestibulum a ornare nunc feugiat. Mauris et risus arcu. Cras dictum
porta cursus. Donec tempus laoreet eros. Nam nec turpis non dui
hendrerit laoreet eu ut ipsum. Nam in sem eget turpis lacinia euismod
eu eget nulla.

Suspendisse at varius elit. Donec consectetur pharetra massa nec
viverra. Cras vehicula lorem id sapien hendrerit tristique. Mauris
vitae mi ipsum. Suspendisse feugiat commodo iaculis. Maecenas vitae
dignissim nunc. Sed hendrerit, arcu et aliquet suscipit, urna quam
fermentum eros, vel accumsan metus quam quis risus. Praesent id eros
pulvinar tellus fringilla cursus. Sed nec vulputate ipsum. Suspendisse
sagittis, magna vitae faucibus semper, nibh felis vehicula tortor, et
molestie velit lorem ac massa.

Duis aliquam accumsan lobortis. Morbi interdum cursus risus, vel
dapibus nisl fermentum sit amet. Etiam in mauris at lectus lacinia
mollis. Proin pretium sem nibh, id scelerisque arcu. Mauris pretium
adipiscing metus. Suspendisse quis convallis augue. Aliquam sed dui
augue, vel tempor ligula. Suspendisse luctus velit quis urna suscipit
sit amet ullamcorper nunc mollis. Praesent vitae velit justo. Donec
quis risus felis. Nullam rutrum, odio non varius ornare, tortor odio
posuere felis, eget accumsan sem sapien et nunc. Fusce mi neque,
elementum non convallis eu, hendrerit id arcu. Morbi tempus tincidunt
ullamcorper. Nullam blandit, diam quis sollicitudin tincidunt, elit
justo varius lacus, aliquet luctus neque nibh quis turpis. Etiam massa
sapien, tristique ut consectetur eu, elementum vel orci.  " }

fn input3() -> str { " Lorem ipsum dolor sit amet, consectetur
adipiscing elit. Pellentesque bibendum sapien ut magna fringilla
mollis. Vivamus in neque non metus faucibus accumsan eu pretium
nunc. Ut erat augue, pulvinar eget blandit nec, cursus quis
ipsum. Aliquam eu ornare risus. Mauris ipsum tortor, posuere vel
gravida ut, tincidunt eu nunc. Aenean pellentesque, justo eu aliquam
condimentum, neque eros feugiat nibh, in dictum nisi augue euismod
lectus. Nam fringilla placerat metus aliquam rutrum. Nullam dapibus
vehicula ligula ut tempor. Aliquam vehicula, diam vitae fermentum
aliquam, justo augue venenatis enim, porta euismod dolor libero in
arcu. Sed sollicitudin dictum eros non ornare. Donec nec purus
orci. Mauris euismod fringilla consequat. Praesent non erat quis risus
dapibus semper ac adipiscing lorem. Aliquam pulvinar dapibus
mollis. Donec fermentum sollicitudin metus, sit amet condimentum leo
adipiscing a.

Vestibulum mi felis, commodo placerat rhoncus sed, feugiat tincidunt
orci. Integer faucibus ornare placerat. Nam et odio massa. Suspendisse
porttitor nunc quis mi mollis imperdiet. Ut ut neque ipsum, sit amet
facilisis erat. Nam ac lacinia turpis. Vivamus ullamcorper iaculis
odio, et euismod sem imperdiet non. Duis porta felis sit amet nunc
venenatis eu vestibulum nisi scelerisque. Nullam luctus mollis nunc
vel pulvinar. Nam lorem tellus, imperdiet sed sodales eu, auctor ut
nunc.

Nulla at mauris at leo sagittis varius eu a elit. Etiam consequat,
tellus ut sagittis porttitor, est justo convallis eros, quis suscipit
justo tortor vitae sem. In in odio augue. Pellentesque habitant morbi
tristique senectus et netus et malesuada fames ac turpis
egestas. Nulla varius ornare ligula quis euismod. Maecenas lobortis
sodales sapien a mattis. Nulla blandit lobortis lacus, ut lobortis
neque dictum ut. Praesent semper laoreet nisl. Etiam arcu eros,
pretium eget eleifend eu, condimentum quis leo. Donec imperdiet porta
erat. Aenean tempor sapien ut arcu porta mollis. Duis ultrices commodo
quam venenatis commodo.

Aliquam odio tellus, tincidunt nec condimentum pellentesque, semper
eget magna. Nam et lacus urna. Pellentesque urna nisi, pharetra vitae
dignissim non, scelerisque eu massa. Sed sapien neque, cursus a
malesuada ut, porta et quam. Donec odio sapien, blandit non aliquam
vel, lobortis quis ligula. Nullam fermentum velit nec quam ultrices et
venenatis sapien congue. Pellentesque vitae nunc arcu. Nullam eget
laoreet nulla. Curabitur dignissim convallis nunc sed blandit. Sed ac
ipsum mi. Ut euismod tellus hendrerit arcu egestas sollicitudin. Nam
eget laoreet ipsum. Morbi sed nulla odio, at volutpat ante. Vivamus
elementum dictum gravida.

Phasellus diam nisi, ullamcorper et placerat non, ultrices ut
lectus. Etiam tincidunt scelerisque imperdiet. Quisque pretium pretium
urna quis cursus. Sed sit amet velit sem. Maecenas eu orci et leo
ultricies dictum. Mauris pellentesque ante a purus gravida
convallis. Integer non tellus ante. Nulla hendrerit lobortis augue sit
amet vulputate. Donec cursus hendrerit diam convallis
luctus. Curabitur ipsum mauris, fermentum quis tincidunt ac, laoreet
sollicitudin sapien. Fusce velit urna, gravida non pulvinar eu, tempor
id nunc.  " }