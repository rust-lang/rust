import std.io;
import std._vec;
import std._str;

tag boxtype {box_h; box_v; box_hv; box_align;}
tag contexttype {cx_h; cx_v;}

tag token {
  brk(uint);
  word(str);
  cword(str); // closing token
  open(boxtype, uint);
  close;
}

type context = rec(contexttype tp, uint indent);

type ps = @rec(mutable vec[context] context,
               uint width,
               io.writer out,
               mutable vec[token] buffered,
               mutable uint scandepth,
               mutable uint bufferedcol,
               mutable uint col,
               mutable bool start_of_line);

fn mkstate(io.writer out, uint width) -> ps {
  let vec[context] stack = vec(rec(tp=cx_v, indent=0u));
  let vec[token] buff = vec();
  ret @rec(mutable context=stack,
           width=width,
           out=out,
           mutable buffered=buff,
           mutable scandepth=0u,
           mutable bufferedcol=0u,
           mutable col=0u,
           mutable start_of_line=true);
}

impure fn push_context(ps p, contexttype tp, uint indent) {
  before_print(p, false);
  p.context = _vec.push[context](p.context, rec(tp=tp, indent=base_indent(p)
                                                + indent));
}

impure fn pop_context(ps p) {
  p.context = _vec.pop[context](p.context);
}

impure fn add_token(ps p, token tok) {
  if (p.width == 0u) {direct_token(p, tok);}
  else if (p.scandepth == 0u) {do_token(p, tok);}
  else {buffer_token(p, tok);}
}

impure fn direct_token(ps p, token tok) {
  alt (tok) {
    case (brk(?sz)) {
      while (sz > 0u) {p.out.write_str(" "); sz -= 1u;}
    }
    case (word(?w)) {p.out.write_str(w);}
    case (cword(?w)) {p.out.write_str(w);}
    case (_) {}
  }
}

impure fn buffer_token(ps p, token tok) {
  p.buffered += vec(tok);
  p.bufferedcol += token_size(tok);
  alt (p.buffered.(0)) {
    case (brk(_)) {
      alt (tok) {
        case (brk(_)) {
          if (p.scandepth == 1u) {finish_break_scan(p);}
        }
        case (open(box_h,_)) {p.scandepth += 1u;}
        case (open(_,_)) {finish_break_scan(p);}
        case (close) {
          p.scandepth -= 1u;
          if (p.scandepth == 0u) {finish_break_scan(p);}
        }
        case (_) {}
      }
    }
    case (open(_,_)) {
      if (p.bufferedcol > p.width) {finish_block_scan(p, cx_v);}
      else {
        alt (tok) {
          case (open(_,_)) {p.scandepth += 1u;}
          case (close) {
            p.scandepth -= 1u;
            if (p.scandepth == 0u) {finish_block_scan(p, cx_h);}
          }
          case (_) {}
        }
      }
    }
  }
}

impure fn finish_block_scan(ps p, contexttype tp) {
  auto indent;
  alt (p.buffered.(0)){
    case (open(box_hv,?ind)) {
      indent = ind;
    }
    case (open(box_align, _)) {
      indent = p.col - base_indent(p);
    }
  }
  p.scandepth = 0u;
  push_context(p, tp, indent);
  for (token t in _vec.shift[token](p.buffered)) {add_token(p, t);}
}

impure fn finish_break_scan(ps p) {
  if (p.bufferedcol > p.width) {
    line_break(p);
  }
  else {
    auto width;
    alt (p.buffered.(0)) {case(brk(?w)) {width = w;}}
    auto i = 0u;
    while (i < width) {p.out.write_str(" "); i+=1u;}
    p.col += width;
  }
  p.scandepth = 0u;
  for (token t in _vec.shift[token](p.buffered)) {add_token(p, t);}
}

impure fn start_scan(ps p, token tok) {
  p.buffered = vec(tok);
  p.scandepth = 1u;
  p.bufferedcol = p.col;
}

fn cur_context(ps p) -> context {
  ret p.context.(_vec.len[context](p.context)-1u);
}
fn base_indent(ps p) -> uint {
  auto i = _vec.len[context](p.context);
  while (i > 0u) {
    i -= 1u;
    auto cx = p.context.(i);
    if (cx.tp == cx_v) {ret cx.indent;}
  }
}

impure fn do_token(ps p, token tok) {
  alt (tok) {
    case (brk(?sz)) {
      alt (cur_context(p).tp) {
        case (cx_h) {
          before_print(p, false);
          start_scan(p, tok);
        }
        case (cx_v) {
          line_break(p);
        }
      }
    }
    case (word(?w)) {
      before_print(p, false);
      p.out.write_str(w);
      p.col += _str.byte_len(w); // TODO char_len
    }
    case (cword(?w)) {
      before_print(p, true);
      p.out.write_str(w);
      p.col += _str.byte_len(w); // TODO char_len
    }
    case (open(?tp, ?indent)) {
      alt (tp) {
        case (box_hv) {start_scan(p, tok);}
        case (box_align) {start_scan(p, tok);}
        case (box_h) {push_context(p, cx_h, indent);}
        case (box_v) {push_context(p, cx_v, indent);}
      }
    }
    case (close) {pop_context(p);}
  }
}

impure fn line_break(ps p) {
  p.out.write_str("\n");
  p.col = 0u;
  p.start_of_line = true;
}

impure fn before_print(ps p, bool closing) {
  if (p.start_of_line) {
    p.start_of_line = false;
    auto ind;
    if (closing) {ind = base_indent(p);}
    else {ind = cur_context(p).indent;}
    p.col = ind;
    while (ind > 0u) {p.out.write_str(" "); ind -= 1u;}
  }
}

fn token_size(token tok) -> uint {
  alt (tok) {
    case (brk(?sz)) {ret sz;}
    case (word(?w)) {ret _str.byte_len(w);}
    case (cword(?w)) {ret _str.byte_len(w);}
    case (open(_, _)) {ret 0u;} // TODO exception for V blocks?
    case (close) {ret 0u;}
  }
}

impure fn box(ps p, uint indent) {add_token(p, open(box_hv, indent));}
impure fn abox(ps p) {add_token(p, open(box_align, 0u));}
impure fn vbox(ps p, uint indent) {add_token(p, open(box_v, indent));}
impure fn hbox(ps p, uint indent) {add_token(p, open(box_h, indent));}
impure fn end(ps p) {add_token(p, close);}
impure fn wrd(ps p, str wrd) {add_token(p, word(wrd));}
impure fn cwrd(ps p, str wrd) {add_token(p, cword(wrd));}
impure fn space(ps p) {add_token(p, brk(1u));}
impure fn spaces(ps p, uint n) {add_token(p, brk(n));}
impure fn line(ps p) {add_token(p, brk(0u));}
