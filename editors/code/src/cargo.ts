import { window } from 'vscode';
import * as cp from 'child_process';
import * as readline from 'readline';

interface CompilationArtifact {
    fileName: string;
    name: string;
    kind: string;
    isTest: boolean;
}

export class Cargo {
    rootFolder: string;
    env?: { [key: string]: string };

    public constructor(cargoTomlFolder: string) {
        this.rootFolder = cargoTomlFolder;
    }

    public async artifactsFromArgs(cargoArgs: string[]): Promise<CompilationArtifact[]> {
        let artifacts: CompilationArtifact[] = [];

        try {
            await this.runCargo(cargoArgs,
                message => {
                    if (message.reason == 'compiler-artifact' && message.executable) {
                        let isBinary = message.target.crate_types.includes('bin');
                        let isBuildScript = message.target.kind.includes('custom-build');
                        if ((isBinary && !isBuildScript) || message.profile.test) {
                            artifacts.push({
                                fileName: message.executable,
                                name: message.target.name,
                                kind: message.target.kind[0],
                                isTest: message.profile.test
                            })
                        }
                    }
                },
                _stderr => {
                    // TODO: to output
                }
            );
        }
        catch (err) {
            // TODO: to output
            throw new Error(`Cargo invocation has failed: ${err}`);
        }

        return artifacts;
    }

    public async executableFromArgs(cargoArgs: string[], extraArgs?: string[]): Promise<string> {
        cargoArgs.push("--message-format=json");
        if (extraArgs) {
            cargoArgs.push('--');
            cargoArgs.push(...extraArgs);
        }

        let artifacts = await this.artifactsFromArgs(cargoArgs);

        if (artifacts.length == 0 ) {
            throw new Error('No compilation artifacts');
        } else if (artifacts.length > 1) {
            throw new Error('Multiple compilation artifacts are not supported.');
        }

        return artifacts[0].fileName;
    }

    runCargo(
        cargoArgs: string[],
        onStdoutJson: (obj: any) => void,
        onStderrString: (data: string) => void
    ): Promise<number> {
        return new Promise<number>((resolve, reject) => {
            let cargo = cp.spawn('cargo', cargoArgs, {
                stdio: ['ignore', 'pipe', 'pipe'],
                cwd: this.rootFolder,
                env: this.env,
            });

            cargo.on('error', err => {
                reject(new Error(`could not launch cargo: ${err}`));
            });
            cargo.stderr.on('data', chunk => {
                onStderrString(chunk.toString());
            });

            let rl = readline.createInterface({ input: cargo.stdout });
            rl.on('line', line => {
                let message = JSON.parse(line);
                onStdoutJson(message);
            });

            cargo.on('exit', (exitCode, _) => {
                if (exitCode == 0)
                    resolve(exitCode);
                else
                    reject(new Error(`exit code: ${exitCode}.`));
            });
        });
    }
}