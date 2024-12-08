# github-release

An action used to publish GitHub releases for `wasmtime`.

As of the time of this writing there's a few actions floating around which
perform github releases but they all tend to have their set of drawbacks.
Additionally nothing handles deleting releases which we need for our rolling
`dev` release.

To handle all this this action rolls-its-own implementation using the
actions/toolkit repository and packages published there. These run in a Docker
container and take various inputs to orchestrate the release from the build.

More comments can be found in `main.js`.

Testing this is really hard. If you want to try though run `npm install` and
then `node main.js`. You'll have to configure a bunch of env vars though to get
anything reasonably working.
