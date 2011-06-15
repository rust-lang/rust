


// -*- rust -*-
tag color {
    rgb(int, int, int);
    rgba(int, int, int, int);
    hsl(int, int, int);
}

fn process(color c) -> int {
    let int x;
    alt (c) {
        case (rgb(?r, _, _)) { log "rgb"; log r; x = r; }
        case (rgba(_, _, _, ?a)) { log "rgba"; log a; x = a; }
        case (hsl(_, ?s, _)) { log "hsl"; log s; x = s; }
    }
    ret x;
}

fn main() {
    let color gray = rgb(127, 127, 127);
    let color clear = rgba(50, 150, 250, 0);
    let color red = hsl(0, 255, 255);
    assert (process(gray) == 127);
    assert (process(clear) == 0);
    assert (process(red) == 255);
}