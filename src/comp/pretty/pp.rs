import std.IO;
import std.Vec;
import std.Str;

tag boxtype {box_h; box_v; box_hv; box_align;}
tag contexttype {cx_h; cx_v;}
tag scantype {scan_hv; scan_h; scan_none;}

tag token {
  brk(uint);
  hardbrk;
  word(str);
  cword(str); // closing token
  open(boxtype, uint);
  close;
}

type context = rec(contexttype tp, uint indent);

type ps = @rec(mutable vec[context] context,
               uint width,
               IO.writer out,
               mutable uint col,
               mutable uint spaces,
               mutable vec[token] buffered,
               mutable scantype scanning,
               mutable vec[boxtype] scandepth,
               mutable uint scancol,
               mutable bool start_of_line,
               mutable bool start_of_box,
               mutable bool potential_brk);

fn mkstate(IO.writer out, uint width) -> ps {
  let vec[context] stack = vec(rec(tp=cx_v, indent=0u));
  let vec[token] buff = vec();
  let vec[boxtype] sd = vec();
  ret @rec(mutable context=stack,
           width=width,
           out=out,
           mutable col=0u,
           mutable spaces=0u,
           mutable buffered=buff,
           mutable scanning=scan_none,
           mutable scandepth=sd,
           mutable scancol=0u,
           mutable start_of_line=true,
           mutable start_of_box=true,
           mutable potential_brk=false);
}

fn write_spaces(ps p, uint i) {
  while (i > 0u) {
    i -= 1u;
    p.out.write_str(" ");
  }
}

fn push_context(ps p, contexttype tp, uint indent) {
  before_print(p, false);
  Vec.push[context](p.context, rec(tp=tp, indent=indent));
  p.start_of_box = true;
}

fn pop_context(ps p) {
  Vec.pop[context](p.context);
}

fn add_token(ps p, token tok) {
  if (p.width == 0u) {direct_token(p, tok);}
  else if (p.scanning == scan_none) {do_token(p, tok);}
  else {buffer_token(p, tok);}
}

fn direct_token(ps p, token tok) {
  alt (tok) {
    case (brk(?sz)) {write_spaces(p, sz);}
    case (word(?w)) {p.out.write_str(w);}
    case (cword(?w)) {p.out.write_str(w);}
    case (_) {}
  }
}

fn buffer_token(ps p, token tok) {
  p.buffered += vec(tok);
  auto col = p.scancol;
  p.scancol = col + token_size(tok);
  if (p.scancol > p.width) {
    finish_scan(p, false);
  } else {
    alt (tok) {
      case (open(?tp,_)) {
        Vec.push[boxtype](p.scandepth, tp);
        if (p.scanning == scan_h) {
          if (tp == box_h) {
            check_potential_brk(p);
          }
        }
      }
      case (close) {
        Vec.pop[boxtype](p.scandepth);
        if (Vec.len[boxtype](p.scandepth) == 0u) {
          finish_scan(p, true);
        }
      }
      case (brk(_)) {
        if (p.scanning == scan_h) {
          if (p.scandepth.(Vec.len[boxtype](p.scandepth)-1u) == box_v) {
            finish_scan(p, true);
          }
        }
      }
      case (_) {}
    }
  }
}

fn check_potential_brk(ps p) {
  for (boxtype tp in p.scandepth) {
    if (tp != box_h) {ret;}
  }
  p.potential_brk = true;
}

fn finish_scan(ps p, bool fits) {
  auto buf = p.buffered;
  auto front = Vec.shift[token](buf);
  auto chosen_tp = cx_h;
  if (!fits) {chosen_tp = cx_v;}
  alt (front) {
    case (open(box_hv, ?ind)) {
      push_context(p, chosen_tp, base_indent(p) + ind);
    }
    case (open(box_align, _)) {
      push_context(p, chosen_tp, p.col);
    }
    case (open(box_h, ?ind)) {
      if (!fits && !p.start_of_box && !p.start_of_line && !p.potential_brk) {
        line_break(p);
      }
      push_context(p, cx_h, base_indent(p) + ind);
    }
  }
  p.scandepth = vec();
  p.scanning = scan_none;
  for (token t in buf) { add_token(p, t); }
}

fn start_scan(ps p, token tok, scantype tp) {
  p.buffered = vec();
  p.scancol = p.col;
  p.scanning = tp;
  buffer_token(p, tok);
  p.potential_brk = false;
}

fn cur_context(ps p) -> context {
  ret p.context.(Vec.len[context](p.context)-1u);
}
fn base_indent(ps p) -> uint {
  auto i = Vec.len[context](p.context);
  while (i > 0u) {
    i -= 1u;
    auto cx = p.context.(i);
    if (cx.tp == cx_v) {ret cx.indent;}
  }
}

fn cx_is(contexttype a, contexttype b) -> bool {
  if (a == b) {ret true;}
  else {ret false;}
}
fn box_is(boxtype a, boxtype b) -> bool {
  if (a == b) {ret true;}
  else {ret false;}
}

fn do_token(ps p, token tok) {
  auto start_of_box = p.start_of_box;
  p.start_of_box = false;
  alt (tok) {
    case (brk(?sz)) {
      if (cx_is(cur_context(p).tp, cx_v) || sz + p.col > p.width) {
        line_break(p);
      }
      else {
        p.spaces += sz;
      }
    }
    case (hardbrk) {
      line_break(p);
    }
    case (word(?w)) {
      auto len = Str.char_len(w);
      if (len + p.col + p.spaces > p.width && !start_of_box &&
          !p.start_of_line) {
        line_break(p);
      }
      before_print(p, false);
      p.out.write_str(w);
      p.col += len;
    }
    case (cword(?w)) {
      before_print(p, true);
      p.out.write_str(w);
      p.col += Str.char_len(w);
    }
    case (open(?tp, ?indent)) {
      if (tp == box_v) {
        push_context(p, cx_v, base_indent(p) + indent);
      } else if (box_is(tp, box_h) && cx_is(cur_context(p).tp, cx_v)) {
        push_context(p, cx_h, base_indent(p) + indent);
      } else if (tp == box_h) {
        p.start_of_box = start_of_box;
        start_scan(p, tok, scan_h);
      } else {
        p.start_of_box = start_of_box;
        start_scan(p, tok, scan_hv);
      }
    }
    case (close) {
      pop_context(p);
    }
  }
}

fn line_break(ps p) {
  p.out.write_str("\n");
  p.col = 0u;
  p.spaces = cur_context(p).indent;
  p.start_of_line = true;
}

fn before_print(ps p, bool closing) {
  if (p.start_of_line) {
    p.start_of_line = false;
    if (closing) {p.spaces = base_indent(p);}
    else {p.spaces = cur_context(p).indent;}
  }
  if (p.spaces > 0u) {
    write_spaces(p, p.spaces);
    p.col += p.spaces;
    p.spaces = 0u;
  }
}

fn token_size(token tok) -> uint {
  alt (tok) {
    case (brk(?sz)) {ret sz;}
    case (hardbrk) {ret 0xFFFFFFu;}
    case (word(?w)) {ret Str.char_len(w);}
    case (cword(?w)) {ret Str.char_len(w);}
    case (open(_, _)) {ret 0u;}
    case (close) {ret 0u;}
  }
}

fn box(ps p, uint indent) {add_token(p, open(box_hv, indent));}
fn abox(ps p) {add_token(p, open(box_align, 0u));}
fn vbox(ps p, uint indent) {add_token(p, open(box_v, indent));}
fn hbox(ps p, uint indent) {add_token(p, open(box_h, indent));}
fn end(ps p) {add_token(p, close);}
fn wrd(ps p, str wrd) {add_token(p, word(wrd));}
fn cwrd(ps p, str wrd) {add_token(p, cword(wrd));}
fn space(ps p) {add_token(p, brk(1u));}
fn spaces(ps p, uint n) {add_token(p, brk(n));}
fn line(ps p) {add_token(p, brk(0u));}
fn hardbreak(ps p) {add_token(p, hardbrk);}
