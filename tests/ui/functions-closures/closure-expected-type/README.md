Some tests targeted at how we deduce the types of closure arguments.
This process is a result of some heuristics aimed at combining the
*expected type* we have with the *actual types* that we get from
inputs. This investigation was kicked off by #38714, which revealed
some pretty deep flaws in the ad-hoc way that we were doing things
before.

See also `src/test/ui/closure-expected-type`.
